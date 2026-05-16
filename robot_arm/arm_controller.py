import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String,Int8
from geometry_msgs.msg import WrenchStamped,Pose, Point, Quaternion
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
import threading
import json
import time
import os
import tkinter as tk
from collections import deque
L1, L2, L3 = 0.28787, 0.26096, 0.26136
D6 = 0.07074
J4_OFFSET_Y = 0.02175
A1=0.020885

Q_MIN = np.radians([-180.0, -180.0, -120.0, -180.0, -100.0, -360.0])
Q_MAX = np.radians([ 180.0,  30.0,  150.0,  180.0,  100.0,  360.0])
print(os.environ.get("ROS_DOMAIN_ID", 0))
# กำหนดขีดจำกัดของรางสไลด์ (หน่วยเป็นเมตร)
SLIDER_MIN = -1.0
SLIDER_MAX = 1.0

# Set to False to disable all joint/slider limit enforcement (for testing only)
JOINT_LIMITS_ENABLED = False
# ========================================================

class RobotVelocityKinematics:
    def __init__(self):
        pass

    def _dh_matrix(self, theta, d, a, alpha):
        return np.array([
            [np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
            [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
            [0,             np.sin(alpha),               np.cos(alpha),              d],
            [0,             0,                           0,                          1]
        ])

    def get_jacobian(self, q):
        l1, l2, l3 = L1, L2, L3
        d6 = D6
        offset_y = J4_OFFSET_Y
        a1 = A1
        
        gamma = np.arctan2(offset_y, l3)
        l3_eff = np.sqrt(l3**2 + offset_y**2)
        q1,q2,q3,q4,q5,q6 = q
        
        dh_table = [
            [q1,           l1, a1,  -np.pi/2], 
            [q2,           0,  l2,  0      ], 
            [q3 + np.pi/2 - gamma, 0,  0,   np.pi/2], 
            [q4,           l3_eff, 0,  -np.pi/2], 
            [q5+gamma,     0,  0,   np.pi/2], 
            [q6,           d6, 0,   0      ]  
        ]

        T_matrices = [np.eye(4)]
        T = np.eye(4)
        for row in dh_table:
            T = T @ self._dh_matrix(*row)
            T_matrices.append(T)

        p_e = T_matrices[-1][0:3, 3]
        J = np.zeros((6, 6))

        for i in range(6):
            z_i = T_matrices[i][0:3, 2]  
            p_i = T_matrices[i][0:3, 3]  
            J[0:3, i] = np.cross(z_i, (p_e - p_i))
            J[3:6, i] = z_i

        return J

    def forward_velocity(self, q, q_dot):
        J = self.get_jacobian(q)
        return J @ np.array(q_dot)

    def inverse_velocity(self, q, target_velocity):
        J = self.get_jacobian(q)
        J_pinv = np.linalg.pinv(J)
        return J_pinv @ np.array(target_velocity)

# ================= ROS2 Node =================
class JointPublisher(Node):
    def __init__(self):
        super().__init__('ik_joint_publisher')
        self.action = self.create_subscription(String,'/arm_command',self.actioncb,10)
        self.publisher = self.create_publisher(JointState, '/motor', 10) #convert to /motor
        self.subscription = self.create_subscription(JointState, '/motor', self.joint_cb, 10) #convert to /joint_state
        self.tasksub = self.create_subscription(String, '/goto_position', self.taskcb, 10)
        self.force_sub = self.create_subscription(WrenchStamped, '/force_sensor', self.force_cb, 10)
        self.current_action = self.create_subscription(String, '/current_action', self.current_action, 10)
        self.safety_status = 0
        self.robot_status = 0
        self.last_act = ""
        self.safety_status_sub = self.create_subscription(Int8, '/safety_status', self.camsafe_cb, 10)
        self.robot_status_sub = self.create_subscription(Int8, '/robot_status', self.botstatus_cb, 10)
        
        self.machine_state_pub = self.create_publisher(Int8, '/machine_state', 10)
        self.ee_pose_pub = self.create_publisher(Pose, '/ee_pose', 10)
        self.goto_pub = self.create_publisher(String, '/goto_position', 10)
        self.current_machine_state= 0
        self.current_joint_positions = [0.0] * 6
        self.current_slider_position = 0.0
        self.has_received_data = False

        self.joints_to_publish =[0.0, -1.570796,1.570796, 0.0, 0.0, 0.0]
        self.rail_to_publish = 0.0
        self.velocity_to_publish = [0.0] * 7 
        self.timer = self.create_timer(0.02, self.timer_callback)
        
        self.kinematics = RobotVelocityKinematics()
        self.is_force_moving = False
        self.is_busy = False
        self.stop_event = threading.Event()

        # Trail visualization — one Path publisher per FK frame (base + joint_1..6)
        self._trail_names = ['base', 'joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'ee']
        self._trail_max = 800  # ~16 s of history at 50 Hz
        self._trails = [deque(maxlen=self._trail_max) for _ in range(7)]  # stores (x,y,z,qx,qy,qz,qw)
        self._trail_distances = [0.0] * 7       # total path length per frame (metres)
        self._trail_prev_pos = [None] * 7       # previous position for delta calc
        self._path_pubs = [
            self.create_publisher(Path, f'/joint_trails/{n}', 10)
            for n in self._trail_names
        ]
        self._text_pub = self.create_publisher(MarkerArray, '/joint_trail_info', 10)
        self._trail_colors = [
            (1.0, 0.2, 0.2),  # base     – red
            (1.0, 0.6, 0.0),  # joint_1  – orange
            (1.0, 1.0, 0.0),  # joint_2  – yellow
            (0.2, 1.0, 0.2),  # joint_3  – green
            (0.0, 0.8, 1.0),  # joint_4  – cyan
            (0.4, 0.4, 1.0),  # joint_5  – blue
            (1.0, 0.2, 1.0),  # EE       – magenta
        ]
        
    def _publish_trails(self, t_list):
        stamp = self.get_clock().now().to_msg()
        text_ma = MarkerArray()

        for i, T in enumerate(t_list):
            pos = T[:3, 3]
            qx, qy, qz, qw = _rot_to_quat(T[:3, :3])

            # accumulate path length since last sample
            if self._trail_prev_pos[i] is not None:
                self._trail_distances[i] += float(np.linalg.norm(pos - self._trail_prev_pos[i]))
            self._trail_prev_pos[i] = pos.copy()
            self._trails[i].append((float(pos[0]), float(pos[1]), float(pos[2]),
                                    float(qx), float(qy), float(qz), float(qw)))

            # --- Path: full trajectory of this frame's origin (with orientation) ---
            path_msg = Path()
            path_msg.header.frame_id = 'world'   # match your RViz fixed frame
            path_msg.header.stamp = stamp
            for s in self._trails[i]:
                ps = PoseStamped()
                ps.header.frame_id = 'world'
                ps.header.stamp = stamp
                ps.pose.position.x = s[0]
                ps.pose.position.y = s[1]
                ps.pose.position.z = s[2]
                ps.pose.orientation.x = s[3]
                ps.pose.orientation.y = s[4]
                ps.pose.orientation.z = s[5]
                ps.pose.orientation.w = s[6]
                path_msg.poses.append(ps)
            self._path_pubs[i].publish(path_msg)

            color = self._trail_colors[i]
            name  = self._trail_names[i]
            pt    = (float(pos[0]), float(pos[1]), float(pos[2]))

            # --- SPHERE: current position of this frame's origin ---
            sphere = Marker()
            sphere.header.frame_id = 'world'
            sphere.header.stamp = stamp
            sphere.ns = 'trail_dot'
            sphere.id = i
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose.position.x, sphere.pose.position.y, sphere.pose.position.z = pt
            sphere.pose.orientation.w = 1.0
            sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.012
            sphere.color.r, sphere.color.g, sphere.color.b = color
            sphere.color.a = 1.0
            text_ma.markers.append(sphere)

            # --- TEXT: frame name + total distance travelled ---
            text = Marker()
            text.header.frame_id = 'world'
            text.header.stamp = stamp
            text.ns = 'trail_dist'
            text.id = i + 7
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x = pt[0]
            text.pose.position.y = pt[1]
            text.pose.position.z = pt[2] + 0.05   # float above the dot
            text.pose.orientation.w = 1.0
            text.scale.z = 0.022                   # text height (m)
            text.color.r, text.color.g, text.color.b = color
            text.color.a = 1.0
            dist_cm = self._trail_distances[i] * 100.0
            text.text = f'{name}: {dist_cm:.1f} cm'
            text_ma.markers.append(text)

        self._text_pub.publish(text_ma)

    def actioncb(self, msg: String):
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError as e:
            print(f"[actioncb] JSON parse error: {e}  raw={msg.data!r}")
            return

        action = data.get("action", "")
        step   = data.get("step", 0)

        if not action:
            print("[actioncb] Missing 'action' field")
            return

        print(f"[actioncb] step={step}  action={action}")

    def taskcb(self, msg: String):
        if self.current_machine_state == 1:
            print("⚠️ มองข้ามคำสั่งใหม่: หุ่นยนต์กำลังทำงาน (working) อยู่")
            return
        self.task = msg.data
        threading.Thread(target=run_pose, args=(self.task,), daemon=True).start()
   
    def publish_joints(self, joints, base_y):
        # [ADDED JOINT LIMITS] บังคับตำแหน่งที่รับเข้ามาไม่ให้เกินขีดจำกัด
        safe_joints = np.clip(joints, Q_MIN, Q_MAX) if JOINT_LIMITS_ENABLED else np.array(joints, dtype=float)
        safe_base_y = np.clip(base_y, SLIDER_MIN, SLIDER_MAX) if JOINT_LIMITS_ENABLED else float(base_y)
        self.joints_to_publish = [float(q) for q in safe_joints]
        self.rail_to_publish = float(safe_base_y)

    def publish_joints_velo(self, joints_vel, slider_vel=0.0):
       
        dt = 0.02
        safe_slider_vel = slider_vel
        next_slider = self.rail_to_publish + slider_vel * dt
        if JOINT_LIMITS_ENABLED:
            if (next_slider <= SLIDER_MIN and slider_vel < 0) or (next_slider >= SLIDER_MAX and slider_vel > 0):
                safe_slider_vel = 0.0

        safe_joints_vel = list(joints_vel)
        for i in range(6):
            next_q = self.joints_to_publish[i] + joints_vel[i] * dt
            if JOINT_LIMITS_ENABLED:
                if (next_q <= Q_MIN[i] and joints_vel[i] < 0) or (next_q >= Q_MAX[i] and joints_vel[i] > 0):
                    safe_joints_vel[i] = 0.0
        self.velocity_to_publish = [float(safe_slider_vel)] + [float(q) for q in safe_joints_vel]

    def machine_state(self, stat):
        out_msg = Int8()
        self.current_machine_state = stat
        out_msg.data = stat
        print(self.current_machine_state)
        self.machine_state_pub.publish(out_msg)

    def ee_pose(self, pose):
        out_msg = Pose()
        pos_x, pos_y, pos_z = pose[:3]
        roll, pitch, yaw = pose[3:]
        out_msg.position = Point(x=float(pos_x), y=float(pos_y), z=float(pos_z))
        qx, qy, qz, qw = rpy_degrees_to_quaternion(roll, pitch, yaw)
        out_msg.orientation = Quaternion(x=float(qx), y=float(qy), z=float(qz), w=float(qw))
        self.ee_pose_pub.publish(out_msg)

    def publish_goto(self, joints_deg, rail_mm, speed_pct=50, mode='effector'):
        payload = [{'label': 'move_point', 'sequence': 1,
                    'joint_1': float(joints_deg[0]), 'joint_2': float(joints_deg[1]),
                    'joint_3': float(joints_deg[2]), 'joint_4': float(joints_deg[3]),
                    'joint_5': float(joints_deg[4]), 'joint_6': float(joints_deg[5]),
                    'rail': float(rail_mm), 'speed': int(speed_pct), 'delay': None,
                    'gripper': 0, 'control_mode': mode}]
        msg = String()
        msg.data = json.dumps(payload)
        self.goto_pub.publish(msg)

    def joint_cb(self, msg):
        if len(msg.position) >= 7:
            self.current_slider_position = msg.position[0]
            self.current_joint_positions = list(msg.position[1:7])
            self.has_received_data = True
            
            if not any(self.joints_to_publish): 
                self.joints_to_publish = list(msg.position[1:7])
                self.rail_to_publish = msg.position[0]

    def force_cb(self, msg: WrenchStamped):
        if not self.has_received_data:
            return

        f_x = msg.wrench.force.x
        f_y = msg.wrench.force.y
        f_z = msg.wrench.force.z
        t_x = msg.wrench.torque.x
        t_y = msg.wrench.torque.y
        t_z = msg.wrench.torque.z
        force_vector_ee = np.array([f_x, f_y, f_z, t_x, t_y, t_z])

        if np.linalg.norm(force_vector_ee) < 0.05:
            if self.is_force_moving:
                self.publish_joints_velo([0.0]*6, 0.0)
                self.is_force_moving = False
            return

        self.is_force_moving = True

        K_linear = 0.02  
        K_angular = 0.05 
        # [เพิ่มพารามิเตอร์] ความเร็วในการหันหน้าแกน Z เข้าหาทิศทางการเดิน
        K_align = 0.5 

        V_target_linear_ee = force_vector_ee[0:3] * K_linear
        V_target_angular_ee = force_vector_ee[3:6] * K_angular

        current_q = np.array(self.current_joint_positions)[:6]
        current_slider = self.current_slider_position

        T_cur_list = forward_kinematics_matrices(current_q, L1, L2, L3, D6, current_slider)
        T_ee = T_cur_list[-1]
        R_ee = T_ee[:3, :3] 

        # 1. คำนวณความเร็วเชิงเส้นใน Base Frame
        V_target_linear_base = R_ee @ V_target_linear_ee
        
        # 2. คำนวณความเร็วเชิงมุมใน Base Frame (จากค่า Torque เดิม)
        V_target_angular_base = R_ee @ V_target_angular_ee

        # ================= ส่วนที่เพิ่มใหม่ =================
        # หาขนาดความเร็วบนระนาบ XY
        v_norm_xy = np.linalg.norm(V_target_linear_base[0:2]) 
        
        # ถ้ามีการเคลื่อนที่พอสมควร ให้เริ่มหันหัว
        if v_norm_xy > 0.001: 
            # สร้างเวกเตอร์เป้าหมาย (หันตามทิศที่กำลังไป และ Z=0 เพื่อให้ขนานกับ XY)
            Z_tar = np.array([
                V_target_linear_base[0] / v_norm_xy,
                V_target_linear_base[1] / v_norm_xy,
                0.0
            ])
            
            # ดึงเวกเตอร์แกน Z ปัจจุบันของ End-Effector (มาจาก Column ที่ 3 ของ R_ee)
            Z_cur = R_ee[:, 2]
            
            # ใช้ Cross Product หาแกนและทิศทางการหมุนที่ต้องชดเชย
            omega_align = np.cross(Z_cur, Z_tar)
            
            # บวกค่าความเร็วเชิงมุมที่คำนวณได้เข้าไป (ยิ่งมุมต่างกันมาก ยิ่งหมุนเร็ว)
            V_target_angular_base += K_align * omega_align
        # ===============================================

        V_target_base = np.concatenate((V_target_linear_base, V_target_angular_base))
        J = self.kinematics.get_jacobian(current_q)
        lambda_sq = 0.01
        J_pinv = J.T @ np.linalg.inv(J @ J.T + lambda_sq * np.eye(6))
        q_dot = J_pinv @ V_target_base
        
        max_q_dot = 1.0 
        q_dot = np.clip(q_dot, -max_q_dot, max_q_dot)
        # self.publish_joints_velo(q_dot.tolist(), 0.0)
        
    def current_action(self,action:String):
        current_act = action.data
        if current_act == self.last_act:
            pass
        else:
            node.machine_state(0)
            self.last_act = current_act

    def camsafe_cb(self,msg:Int8):
        self.safety_status = msg.data

    def botstatus_cb(self,msg:Int8):
        self.robot_status = msg.data

    def timer_callback(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = ['slider_joint', 'joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']
        
        dt = 0.02
        if any(v != 0.0 for v in self.velocity_to_publish):
            self.rail_to_publish += self.velocity_to_publish[0] * dt
            if JOINT_LIMITS_ENABLED:
                self.rail_to_publish = np.clip(self.rail_to_publish, SLIDER_MIN, SLIDER_MAX)

            for i in range(6):
                self.joints_to_publish[i] += self.velocity_to_publish[i+1] * dt
                if JOINT_LIMITS_ENABLED:
                    self.joints_to_publish[i] = np.clip(self.joints_to_publish[i], Q_MIN[i], Q_MAX[i])

        msg.position = [float(self.rail_to_publish)] + [float(q) for q in self.joints_to_publish]
        msg.velocity = [float(v) for v in self.velocity_to_publish]
        if self.safety_status ==2 or self.robot_status == 2:
            print("Pause")
        else:
            self.publisher.publish(msg)
        current_T_list = forward_kinematics_matrices(
            self.joints_to_publish, L1, L2, L3, D6, self.rail_to_publish
        )
        current_T = current_T_list[-1]
        current_pose = matrix_to_rpy(current_T)
        self.ee_pose(current_pose)
        self._publish_trails(current_T_list)

# ================= Math & Kinematics =================
def _rot_to_quat(R):
    """3x3 rotation matrix → (qx, qy, qz, qw)."""
    tr = R[0,0] + R[1,1] + R[2,2]
    if tr > 0:
        s = 0.5 / np.sqrt(tr + 1.0)
        return (R[2,1]-R[1,2])*s, (R[0,2]-R[2,0])*s, (R[1,0]-R[0,1])*s, 0.25/s
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        return 0.25*s, (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s, (R[2,1]-R[1,2])/s
    elif R[1,1] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        return (R[0,1]+R[1,0])/s, 0.25*s, (R[1,2]+R[2,1])/s, (R[0,2]-R[2,0])/s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        return (R[0,2]+R[2,0])/s, (R[1,2]+R[2,1])/s, 0.25*s, (R[1,0]-R[0,1])/s

def rpy_to_matrix(roll, pitch, yaw):
    Rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx

def matrix_to_rpy(T):
    x, y, z = T[0, 3], T[1, 3], T[2, 3]
    R = T[:3, :3]
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[0, 0]**2 + R[1, 0]**2))
    
    if np.abs(pitch - np.pi/2) < 1e-6:
        yaw = 0.0
        roll = np.arctan2(R[0, 1], R[0, 2])
    elif np.abs(pitch + np.pi/2) < 1e-6:
        yaw = 0.0
        roll = -np.arctan2(R[0, 1], R[0, 2])
    else:
        yaw = np.arctan2(R[1, 0], R[0, 0])
        roll = np.arctan2(R[2, 1], R[2, 2])
        
    return x, y, z, np.degrees(roll), np.degrees(pitch), np.degrees(yaw)

def get_transform(theta, d, a, alpha):
    return np.array([
        [np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
        [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
        [0,              np.sin(alpha),                np.cos(alpha),               d],
        [0,              0,                            0,                           1]
    ])

def rpy_degrees_to_quaternion(roll_deg, pitch_deg, yaw_deg):
        roll = np.radians(roll_deg)
        pitch = np.radians(pitch_deg)
        yaw = np.radians(yaw_deg)
        qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
        qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
        qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        return qx, qy, qz, qw

def forward_kinematics_matrices(q, l1, l2, l3, d6, base_y=0.0):
    q1, q2, q3, q4, q5, q6 = q
    # gamma accounts for the J4_OFFSET_Y mechanical offset — must match get_jacobian exactly
    gamma  = np.arctan2(J4_OFFSET_Y, l3)
    l3_eff = np.sqrt(l3**2 + J4_OFFSET_Y**2)
    dh_params = [
        [q1,                 l1,     A1,  -np.pi/2],
        [q2,                 0,      l2,   0       ],
        [q3 + np.pi/2 - gamma, 0,    0,   np.pi/2 ],
        [q4,                 l3_eff, 0,  -np.pi/2 ],
        [q5 + gamma,         0,      0,   np.pi/2 ],
        [q6,                 d6,     0,   0        ],
    ]
    T = np.eye(4)
    T[1, 3] = base_y 
    T_list = [T.copy()]
    for params in dh_params:
        T = T @ get_transform(*params)
        T_list.append(T.copy())
    return T_list

def inverse_kinematics_6dof(local_target_pos, target_orient, l1, l2, l3, d6):
    xc, yc, zc = local_target_pos - d6 * target_orient[:, 2]
    a1 = A1
    q1 = np.arctan2(yc, xc)

    r = np.sqrt(xc**2 + yc**2)
    s = zc - l1
    D_sq = r**2 + s**2
    
    cos_q3 = (D_sq - l2**2 - l3**2) / (2 * l2 * l3)
    
    reachable = True
    if cos_q3 > 1.0 or cos_q3 < -1.0:
        reachable = False
        cos_q3 = np.clip(cos_q3, -1.0, 1.0)
        
    sin_q3 = np.sqrt(1 - cos_q3**2) 
    q3 = np.arctan2(sin_q3, cos_q3)

    beta = np.arctan2(l3 * np.sin(q3), l2 + l3 * np.cos(q3))
    q2 = np.arctan2(-s, r) - beta

    T0 = np.eye(4)
    dh03 = [
        [q1, l1, a1, -np.pi/2],
        [q2, 0, l2, 0],
        [q3 + np.pi/2, 0, 0, np.pi/2]
    ]
    for params in dh03:
        T0 = T0 @ get_transform(*params)
    R03 = T0[:3, :3]

    R36 = R03.T @ target_orient
    q5 = np.arctan2(np.sqrt(R36[0,2]**2 + R36[1,2]**2), R36[2,2])

    if np.abs(R36[2, 2]) > 0.9999: #singularlity detected
        q4 = 0
        q6 = np.arctan2(R36[1,0], R36[0,0]) #q6หมุนแทน
    else:
        q4 = np.arctan2(R36[1,2], R36[0,2])
        q6 = np.arctan2(R36[2,1], -R36[2,0])

    return np.array([q1, q2, q3, q4, q5, q6]), reachable

def apply_circular_avoidance(current_pos, target_pos, r_avoid=0.22, z_max=0.4):
    c_x, c_y, c_z = current_pos
    t_x, t_y, t_z = target_pos

    if 0.0 <= t_z <= z_max:
        r_target = np.sqrt(t_x**2 + t_y**2)
        if r_target < r_avoid:
            if r_target > 1e-6:
                scale = r_avoid / r_target
                t_x_new = t_x * scale
                t_y_new = t_y * scale
            else:
                r_curr = np.sqrt(c_x**2 + c_y**2)
                if r_curr > 1e-6:
                    t_x_new = c_x * (r_avoid / r_curr)
                    t_y_new = c_y * (r_avoid / r_curr)
                else:
                    t_x_new = r_avoid 
                    t_y_new = 0.0
            return np.array([t_x_new, t_y_new, t_z]), True 
            
    return target_pos, False 

def reset_home():
    
    tar_q = np.clip([0.0, -1.5708, 1.5708, 0, 0, 0], Q_MIN, Q_MAX)
    speed_pct = 40       
    Kp = 5.0 * (speed_pct / 100.0)
    Kd = 0.1
    prev_error_q = np.zeros(6)
    dt = 0.05
    node.machine_state(1)
    print(f"🚀 [RESET] เคลื่อนที่ด้วย PD Control (Kp: {Kp}, Kd: {Kd})")
    while True:
        current_q = np.array(node.current_joint_positions) 
        error_q = tar_q - current_q
        derivative_q = (error_q - prev_error_q) / dt
        q_dot = (Kp * error_q) + (Kd * derivative_q)

        max_limit = 1.5 
        q_dot = np.clip(q_dot, -max_limit, max_limit)
        reached_mask = np.abs(error_q) < 0.001
        q_dot[reached_mask] = 0.0
        node.publish_joints_velo(q_dot.tolist(),0)

        prev_error_q = error_q
        
        if np.all(reached_mask):
            print("🎯 [RESET] กลับตำแหน่ง Home เรียบร้อยแล้ว!")
            break
        
    node.publish_joints_velo([0.0]*6,0)

def move_save(task):
    tar_q = np.clip(np.radians(task[:6]), Q_MIN, Q_MAX) if JOINT_LIMITS_ENABLED else np.radians(task[:6])
    target_slider = np.clip(task[6]/1000, SLIDER_MIN, SLIDER_MAX) if JOINT_LIMITS_ENABLED else task[6]/1000
    
    speed_pct = task[7] 
    gripper = task[8]        
    node.machine_state(1)
    Kp = 0.5 * (speed_pct / 100.0)
    Kd = 0.1
    dt = 0.05

    # initialize to actual error so first-iteration derivative = 0 (no spike)
    _q0 = np.array(node.current_joint_positions)
    prev_error_q = tar_q - _q0
    prev_error_slider = target_slider - node.current_slider_position

    r_avoid = 0.3   # base exclusion radius (m)

    # pre-flight: cancel if target EE position is inside the base radius
    T_tar_list = forward_kinematics_matrices(tar_q, L1, L2, L3, D6, target_slider)
    tar_p = T_tar_list[-1][:3, 3]
    if np.sqrt(tar_p[0]**2 + tar_p[1]**2) < r_avoid:
        print(f"[ERROR] ยกเลิกคำสั่ง! ตำแหน่งเป้าหมายอยู่ในรัศมีฐานหุ่นยนต์ (r={np.sqrt(tar_p[0]**2+tar_p[1]**2):.3f} m < {r_avoid} m)")
        node.machine_state(0)
        return

    print(f"🚀 [MOVE] เคลื่อนที่ด้วย PD Control (Kp: {Kp}, Kd: {Kd})")

    while True:
        current_q = np.array(node.current_joint_positions) 
        current_slider = node.current_slider_position 

        current_T_list = forward_kinematics_matrices(current_q, L1, L2, L3, D6, current_slider)
        current_T = current_T_list[-1]
        current_pose = matrix_to_rpy(current_T)

        # in-loop safety: stop if EE drifts into the base exclusion zone
        ee_p = current_T[:3, 3]
        if np.sqrt(ee_p[0]**2 + ee_p[1]**2) < r_avoid:
            node.publish_joints_velo([0.0]*6, 0.0)
            print(f"⚠️ [MOVE] หยุดฉุกเฉิน! ปลายแขนเข้าใกล้ฐานหุ่นยนต์ (r={np.sqrt(ee_p[0]**2+ee_p[1]**2):.3f} m)")
            node.machine_state(0)
            return

        error_q = tar_q - current_q
        derivative_q = (error_q - prev_error_q) / dt
        q_dot = (Kp * error_q) + (Kd * derivative_q)
        
        max_limit = 2 
        q_dot = np.clip(q_dot, -max_limit, max_limit)
        reached_mask = np.abs(error_q) < 0.05
        q_dot[reached_mask] = 0.0 
        
        error_slider = target_slider - current_slider
        derivative_slider = (error_slider - prev_error_slider) / dt
        slider_dot = (Kp * error_slider) + (Kd * derivative_slider)
        
        max_slider_limit = 0.5 
        slider_dot = np.clip(slider_dot, -max_slider_limit, max_slider_limit)

        slider_reached = abs(error_slider) < 0.005
        if slider_reached:
            slider_dot = 0.0 
            
        if node.safety_status == 1:
            speed_pct=0.5*task[7]
        else:
            speed_pct = task[7]
            
        if node.safety_status ==2 or node.robot_status ==2:
            while node.safety_status ==2 or node.robot_status ==2:
                print("Pause Temporary")

        node.publish_joints_velo(q_dot.tolist(), float(slider_dot))
        
        prev_error_q = error_q
        prev_error_slider = error_slider
        
        if np.all(reached_mask) and slider_reached:
            print("🎯 [MOVE] ทุกแกนและรางสไลด์เข้าสู่ตำแหน่งเป้าหมายแล้ว!")
            node.machine_state(2)
            break
        
    node.publish_joints_velo([0.0]*6, 0.0)



def _jacobian_move(tar_p, tar_R, target_slider, task):
    """Inverse-Jacobian velocity control with DLS singularity prevention."""
    Kp_pos     = 5.0
    Kp_ori     = 5.0
    dt         = 0.02
    W_THRESH   = 0.02
    LAMBDA_MAX = 0.05

    while True:
        if node.safety_status == 2 or node.robot_status == 2:
            node.publish_joints_velo([0.0]*6, 0.0)
            while node.safety_status == 2 or node.robot_status == 2:
                print("Pause Temporary")
        

        eff_speed  = 0.5 * task[7] if node.safety_status == 1 else task[7]

        current_q      = np.array(node.current_joint_positions)[:6]
        current_slider = node.current_slider_position
        T_cur          = forward_kinematics_matrices(current_q, L1, L2, L3, D6, current_slider)[-1]
        cur_p          = T_cur[:3, 3]
        cur_R          = T_cur[:3, :3]

        e_pos = tar_p - cur_p
        e_ori = 0.5 * (np.cross(cur_R[:, 0], tar_R[:, 0]) +
                       np.cross(cur_R[:, 1], tar_R[:, 1]) +
                       np.cross(cur_R[:, 2], tar_R[:, 2]))

        if np.linalg.norm(e_pos) < 0.005 and np.linalg.norm(e_ori) < 0.02:
            break

        V = np.concatenate((Kp_pos * e_pos, Kp_ori * e_ori))
        max_v = 0.5 * (eff_speed / 100.0)
        if np.linalg.norm(V[:3]) > max_v:
            V[:3] = V[:3] / np.linalg.norm(V[:3]) * max_v

        # DLS inverse Jacobian
        J       = jacobian.get_jacobian(current_q)
        w_manip = np.sqrt(max(0.0, np.linalg.det(J @ J.T)))
        if w_manip < W_THRESH:
            lam_sq = LAMBDA_MAX ** 2 * (1.0 - (w_manip / W_THRESH) ** 2)
            print(f"⚠️  [SINGULARITY] w={w_manip:.4f} λ²={lam_sq:.5f}")
        else:
            lam_sq = 0.0
        J_pinv = J.T @ np.linalg.inv(J @ J.T + lam_sq * np.eye(6))
        q_dot  = np.clip(J_pinv @ V, -1.5, 1.5)

        if JOINT_LIMITS_ENABLED:
            for i in range(6):
                if (current_q[i] <= Q_MIN[i] and q_dot[i] < 0) or \
                   (current_q[i] >= Q_MAX[i] and q_dot[i] > 0):
                    q_dot[i] = 0.0

        e_sl   = target_slider - current_slider
        sl_dot = np.clip((eff_speed / 100.0) * e_sl, -0.2, 0.2)

        node.publish_joints_velo(q_dot.tolist(), float(sl_dot))

    node.publish_joints_velo([0.0]*6, 0.0)


def move_save_ee(task):
    BOX_X = 0.2   # forbidden zone ±x (m)
    BOX_Y = 0.2   # forbidden zone ±y (m)
    MARGIN = 0.1  # clearance outside box for via-point

    tar_q         = np.clip(np.radians(task[:6]), Q_MIN, Q_MAX) if JOINT_LIMITS_ENABLED else np.radians(task[:6])
    target_slider = np.clip(task[6] / 1000.0, SLIDER_MIN, SLIDER_MAX) if JOINT_LIMITS_ENABLED else task[6] / 1000.0

    # FK → target EE pose
    T_tar    = forward_kinematics_matrices(tar_q, L1, L2, L3, D6, target_slider)[-1]
    tar_p    = T_tar[:3, 3].copy()
    tar_R    = T_tar[:3, :3].copy()
    tar_pose = matrix_to_rpy(T_tar)
    print(f"[TARGET EE] X={tar_pose[0]:.4f} Y={tar_pose[1]:.4f} Z={tar_pose[2]:.4f} m | "
          f"Roll={tar_pose[3]:.2f} Pitch={tar_pose[4]:.2f} Yaw={tar_pose[5]:.2f} deg")

    # Cancel if target is inside the forbidden box
    if abs(tar_p[0]) <= BOX_X and abs(tar_p[1]) <= BOX_Y:
        print(f"[ERROR] ยกเลิกคำสั่ง! เป้าหมายอยู่ในพื้นที่หวงห้าม "
              f"(X={tar_p[0]:.3f} Y={tar_p[1]:.3f})")
        node.machine_state(0)
        return

    # Current EE position
    cur_q  = np.array(node.current_joint_positions)[:6]
    cur_p  = forward_kinematics_matrices(cur_q, L1, L2, L3, D6,
                                         node.current_slider_position)[-1][:3, 3].copy()

    # Check if the XY straight-line path passes through the forbidden box
    def _hits_box(p1, p2, n=300):
        for k in range(1, n):
            pt = p1 + (k / n) * (p2 - p1)
            if abs(pt[0]) <= BOX_X and abs(pt[1]) <= BOX_Y:
                return True
        return False

    node.machine_state(1)

    if _hits_box(cur_p[:2], tar_p[:2]):
        # Via-point: push out along the direction perpendicular to the path,
        # placed at the nearest box corner + margin, at the midpoint Z height.
        seg    = tar_p[:2] - cur_p[:2]
        perp   = np.array([-seg[1], seg[0]])            # 90° rotation
        perp  /= np.linalg.norm(perp)                   # normalise
        # Pick the perpendicular direction that points away from origin
        if np.dot(perp, 0.5 * (cur_p[:2] + tar_p[:2])) < 0:
            perp = -perp
        via_xy = perp * (max(BOX_X, BOX_Y) + MARGIN)    # place outside box
        via_p  = np.array([via_xy[0], via_xy[1], 0.5 * (cur_p[2] + tar_p[2])])
        print(f"🚧 [BYPASS] เส้นทางตัดผ่านฐาน! แวะ via-point {np.round(via_p, 3)}")
        _jacobian_move(via_p, tar_R, target_slider, task)
        print(f"✅ [MOVE_EE] ถึง via-point มุ่งหน้าสู่เป้าหมาย")

    print(f"🚀 [MOVE_EE] เคลื่อนที่ไปยังเป้าหมาย")
    _jacobian_move(tar_p, tar_R, target_slider, task)
    print("🎯 [MOVE_EE] เข้าสู่ตำแหน่งสำเร็จ!")
    node.machine_state(2)

def instant_jog_joint(task): # 👈 ใส่พารามิเตอร์ task กลับเข้ามาตรงนี้เพื่อแก้ TypeError
    dt = 0.02
    WATCHDOG_TIMEOUT = 0.5  # เผื่อเวลา Network Jitter นิดหน่อย

    # 1. บันทึก Task แรกที่ส่งมาจาก Thread ลงใน Node ทันที พร้อมประทับเวลาเริ่มต้น
    node.latest_task = task
    node.last_task_time = time.time()

    max_joint_vel = np.radians(30.0)  
    max_slider_vel = 0.05             
    Kp = 5.0                          

    print(f"⚡ [JOG JOINT] เริ่มต้นโหมดต่อเนื่อง (Continuous Watchdog Mode)")

    # บันทึกองศาข้อต่อและ Slider เริ่มต้นเพียงครั้งแรกครั้งเดียว
    current_q = np.array(node.current_joint_positions)[:6]
    current_slider = node.current_slider_position
    
    target_q = current_q.copy()
    target_slider = current_slider

    while True:
        current_time = time.time()

        # อ่านคำสั่งล่าสุดที่อาจถูกอัปเดตจาก Callback ภายนอกเข้ามาระหว่างทาง
        current_task = getattr(node, 'latest_task', None)
        last_task_time = getattr(node, 'last_task_time', 0.0)

        # 🛑 [CHECK TIMEOUT] ถ้าไม่มีคำสั่งใหม่ส่งเข้ามาภายในเวลาที่กำหนด ให้หลุดลูปเพื่อหยุดหุ่นยนต์
        if current_task is None or (current_time - last_task_time) > WATCHDOG_TIMEOUT:
            print(f"🛑 [JOG JOINT] Timeout (> {WATCHDOG_TIMEOUT}s) ไม่ได้รับคำสั่งใหม่ สั่งหยุดแกน")
            break

        if isinstance(current_task, list) and len(current_task) > 0:
            current_task = current_task[0]
            
        axis = current_task.get('axis', '')
        direction = current_task.get('direction', 0)
        speed = current_task.get('speed', 0)

        # ถ้าส่ง direction = 0 หรือ speed = 0 มา แปลว่าสั่งประคองตำแหน่งไว้
        if direction == 0 or speed == 0:
            node.publish_joints_velo([0.0] * 6, 0.0)
            continue

        # ตรวจสอบสถานะความปลอดภัย
        if node.safety_status == 2 or node.robot_status == 2:
            node.publish_joints_velo([0.0] * 6, 0.0) 
            print("Pause Temporary")
            continue

        # คำนวณความเร็วของรอบนี้
        joint_vel = direction * (speed / 100.0) * max_joint_vel
        slider_vel = direction * (speed / 100.0) * max_slider_vel

        # =====================================================================
        # 2. อัปเดตเป้าหมาย (สะสมตำแหน่งไปเรื่อยๆ)
        # =====================================================================
        if axis == 'joint_1':               target_q[0] += joint_vel * dt
        elif axis == 'joint_2':             target_q[1] += joint_vel * dt
        elif axis == 'joint_3':             target_q[2] += joint_vel * dt
        elif axis == 'joint_4':             target_q[3] += joint_vel * dt
        elif axis == 'joint_5':             target_q[4] += joint_vel * dt
        elif axis == 'joint_6':             target_q[5] += joint_vel * dt
        elif axis in ['rail', 'slider']:    target_slider += slider_vel * dt

        # --- ตรวจสอบความปลอดภัย: ป้องกันการชนฐาน (Base Area Collision) ---
        T_list_target = forward_kinematics_matrices(target_q, L1, L2, L3, D6, target_slider)
        t_x, t_y, t_z = T_list_target[-1][:3, 3]
        if -0.2 <= t_x <= 0.2 and -0.2 <= t_y <= 0.2 and 0.0 <= t_z <= 0.4:
            print(f"⚠️ [JOG JOINT WARNING] Arm approaching base area หยุดการเคลื่อนที่")
            break

        # =====================================================================
        # 3. คำนวณ Error เพื่อดึงข้อต่อที่ไม่ได้รับคำสั่งให้ล็อคอยู่กับที่
        # =====================================================================
        current_q = np.array(node.current_joint_positions)[:6]
        current_slider = node.current_slider_position

        e_joint = target_q - current_q
        e_slider = target_slider - current_slider
        
        q_dot = Kp * e_joint
        slider_dot = Kp * e_slider

        # คุมความเร็วไม่ให้กระชากเกินขีดจำกัด
        q_dot = np.clip(q_dot, -0.05, 0.05)
        slider_dot = np.clip(slider_dot, -0.5, 0.5)

        # =====================================================================
        # 4. ตรวจสอบ Joint Limit
        # =====================================================================
        if JOINT_LIMITS_ENABLED:
            limit_hit = False
            for i_joint in range(6):  # เปลี่ยนชื่อตัวแปรลูปไม่ให้ซ้ำกับ 'i' ด้านนอก
                if (target_q[i_joint] <= Q_MIN[i_joint] and q_dot[i_joint] < 0) or \
                   (target_q[i_joint] >= Q_MAX[i_joint] and q_dot[i_joint] > 0):
                    print(f"⚠️ [JOG JOINT WARNING] หยุดแขนกล! Joint {i_joint+1} ชนขีดจำกัดแล้ว")
                    limit_hit = True
                    break
            
            if (target_slider <= SLIDER_MIN and slider_dot < 0) or \
               (target_slider >= SLIDER_MAX and slider_dot > 0):
                print(f"⚠️ [JOG JOINT WARNING] หยุดแขนกล! รางสไลด์ (Slider) ชนขีดจำกัดแล้ว")
                limit_hit = True

            if limit_hit:
                break 

        # ส่งคำสั่งความเร็วให้มอเตอร์
        node.publish_joints_velo(q_dot.tolist(), float(slider_dot))

    # เมื่อหลุดออกจากลูป (Timeout หรือชน Limit) สั่งหยุดสนิททันที
    node.publish_joints_velo([0.0] * 6, 0.0)
    print("🛑 [JOG JOINT] จบการทำงานโหมดควบคุมต่อเนื่อง")

def instant_jog_task(task):
    dt = 0.02
    WATCHDOG_TIMEOUT = 0.25  # เผื่อเวลาสำหรับ Network Jitter สัญญาณดีเลย์

    # 1. ผูกค่าคำสั่งแรกเข้ากับ Node เพื่อเปิดระบบ Watchdog
    node.latest_task = task
    node.last_task_time = time.time()

    # ค่าความเร็วสูงสุด
    max_linear_vel = 0.15                 # 15 cm/s at 100%
    max_angular_vel = np.radians(30)      # 30 deg/s at 100%
    Kp_pos = 5.0 
    Kp_ori = 5.0 

    print("⚡ [JOG TASK] เริ่มต้นโหมดคาร์ทีเซียนแบบต่อเนื่อง (Continuous Watchdog Mode)")

    # =====================================================================
    # บันทึกตำแหน่งเริ่มต้น (Reference) "ครั้งแรกครั้งเดียว" เพื่อใช้คำนวณสะสมพิกัด
    # =====================================================================
    current_q = np.array(node.current_joint_positions)[:6]
    current_slider = node.current_slider_position
    T_start_list = forward_kinematics_matrices(current_q, L1, L2, L3, D6, current_slider)
    
    target_pos = T_start_list[-1][:3, 3].copy()
    target_R = T_start_list[-1][:3, :3].copy() 

    while True:
        current_time = time.time()

        # อ่านคำสั่งปัจจุบันที่ส่งมาจากฝั่ง Network / UI เปลียนแปลงแบบ Real-time
        current_task = getattr(node, 'latest_task', None)
        last_task_time = getattr(node, 'last_task_time', 0.0)

        # 🛑 [CHECK TIMEOUT] หากไม่มีคำสั่งใหม่เข้ามาภายใน 25ms ให้เบรกออกจากลูปทันที
        if current_task is None or (current_time - last_task_time) > WATCHDOG_TIMEOUT:
            print(f"🛑 [JOG TASK] Timeout (> {WATCHDOG_TIMEOUT}s) ไม่ได้รับคำสั่งใหม่ สั่งหยุดเคลื่อนที่")
            break

        # แกะค่าตัวแปรภายในลูป เพื่อรองรับการปรับความเร็วหรือทิศทางแบบ Dynamic
        if isinstance(current_task, list) and len(current_task) > 0:
            current_task = current_task[0]
     
        axis = current_task.get('axis', '')
        direction = current_task.get('direction', 0)
        speed = current_task.get('speed', 0)
     
        tcp_x = current_task.get('tcp_x', 0.0) / 1000.0
        tcp_y = current_task.get('tcp_y', 0.0) / 1000.0
        tcp_z = current_task.get('tcp_z', 0.0) / 1000.0
        tcp_offset = np.array([tcp_x, tcp_y, tcp_z])

        # ถ้าส่งทิศทางหรือความเร็วเป็น 0 ให้สั่งหยุดชั่วคราวและวนลูปประคองตำแหน่งไว้
        if direction == 0 or speed == 0:
            node.publish_joints_velo([0.0] * 6, 0.0)
            continue
     
        # ตรวจสอบสถานะความปลอดภัยของระบบ Robot
        if node.safety_status == 2 or node.robot_status == 2:
            node.publish_joints_velo([0.0] * 6, 0.0)
            print("Pause Temporary")

            continue
     
        # คำนวณความเร็วเชิงเส้นและความเร็วเชิงมุมรอบนี้
        linear_vel = direction * (speed / 100.0) * max_linear_vel
        angular_vel = direction * (speed / 100.0) * max_angular_vel
 
        # อัปเดตพิกัดปัจจุบันจริงของหุ่นยนต์ในรอบนี้
        current_q = np.array(node.current_joint_positions)[:6]
        current_slider = node.current_slider_position
        
        T_cur_list = forward_kinematics_matrices(current_q, L1, L2, L3, D6, current_slider)
        curr_pos = T_cur_list[-1][:3, 3]
        curr_R = T_cur_list[-1][:3, :3]

        # =====================================================================
        # 1. จำลองเป้าหมายใหม่ล่วงหน้า (Next Target) เพื่อเอาไปเช็คความปลอดภัย
        # =====================================================================
        next_target_pos = target_pos.copy()
        next_target_R = target_R.copy()

        if axis == 'x':       next_target_pos[0] += linear_vel * dt
        elif axis == 'y':     next_target_pos[1] += linear_vel * dt
        elif axis == 'z':     next_target_pos[2] += linear_vel * dt
        elif axis == 'roll':  next_target_R = rpy_to_matrix(angular_vel * dt, 0, 0) @ next_target_R
        elif axis == 'pitch': next_target_R = rpy_to_matrix(0, angular_vel * dt, 0) @ next_target_R
        elif axis == 'yaw':   next_target_R = rpy_to_matrix(0, 0, angular_vel * dt) @ next_target_R

        # --- เช็คพื้นที่ต้องห้าม X=[-0.15, 0.15], Y=[-0.15, 0.15] ---
        c_tcp = target_pos + (target_R @ tcp_offset)           
        n_tcp = next_target_pos + (next_target_R @ tcp_offset) 

        in_zone_now = (-0.15 <= c_tcp[0] <= 0.15) and (-0.15 <= c_tcp[1] <= 0.15)
        in_zone_next = (-0.15 <= n_tcp[0] <= 0.15) and (-0.15 <= n_tcp[1] <= 0.15)

        if in_zone_next:
            if in_zone_now:
                dist_now = c_tcp[0]**2 + c_tcp[1]**2
                dist_next = n_tcp[0]**2 + n_tcp[1]**2
                if dist_next < dist_now: 
                    print(f"⚠️ [JOG TASK] ติดอยู่ในเขตต้องห้าม! ต้องกดปุ่ม Jog หันออกเท่านั้น")
                    break
            else:
                print(f"⚠️ [JOG TASK WARNING] ห้ามเข้าพื้นที่หวงห้าม! หยุดการเคลื่อนที่")
                break

        # ผ่านการตรวจสอบความปลอดภัย -> อัปเดตพิกัดเป้าหมายจริง
        target_pos = next_target_pos
        target_R = next_target_R

        # =====================================================================
        # 2. คำนวณ Error ดึงหุ่นยนต์เข้าสู่เป้าหมาย (ล็อก Roll Pitch Yaw)
        # =====================================================================
        e_pos = target_pos - curr_pos
        e_ori = 0.5 * (np.cross(curr_R[:, 0], target_R[:, 0]) + 
                       np.cross(curr_R[:, 1], target_R[:, 1]) + 
                       np.cross(curr_R[:, 2], target_R[:, 2]))
        
        V_target_abs = np.concatenate((Kp_pos * e_pos, Kp_ori * e_ori))

        # =====================================================================
        # 3. แปลงความเร็วเป้าหมายเป็นความเร็วมอเตอร์ด้วย DLS Jacobian
        # =====================================================================
        J = node.kinematics.get_jacobian(current_q)
        det_JJT = np.linalg.det(J @ J.T)
        w = np.sqrt(max(0.0, det_JJT))
        w_threshold = 0.05
        lambda_sq = 0.01 * (1.0 - (w / w_threshold) ** 2) if w < w_threshold else 0.0

        J_pinv = J.T @ np.linalg.inv(J @ J.T + lambda_sq * np.eye(6))
        q_dot = J_pinv @ V_target_abs

        # =====================================================================
        # 4. ตรวจสอบ Joint Limit
        # =====================================================================
        if JOINT_LIMITS_ENABLED:
            limit_hit = False
            for i_joint in range(6):
                if (current_q[i_joint] <= Q_MIN[i_joint] and q_dot[i_joint] < 0) or \
                   (current_q[i_joint] >= Q_MAX[i_joint] and q_dot[i_joint] > 0):
                    print(f"⚠️ [JOG TASK WARNING] หยุดแขนกล! Joint {i_joint+1} ชนขีดจำกัดแล้ว")
                    limit_hit = True
                    break
            if limit_hit:
                break

        # ส่งคำสั่งไปที่มอเตอร์
        node.publish_joints_velo(q_dot.tolist(), 0.0)
 
    # ออกจากลูปทำงาน (เมื่อ Timeout หรือชน Limit) -> สั่งหยุดสนิททันที
    node.publish_joints_velo([0.0] * 6, 0.0)
    print(f"🛑 [JOG TASK] จบการทำงานโหมดควบคุมต่อเนื่องเชิงเส้น")

def move_forward_and_align(dist, max_linear_speed=0.05, max_angular_speed=0.5):
        distance = dist.get('forward')/1000
        print(f"🚀 เริ่มทำงาน: วิ่งเป้าหมาย {distance} ซม. และหมุนขนานพื้นไปพร้อมกัน")

        # =========================================================
        # 1. อ่านค่าเริ่มต้น เพื่อคำนวณเป้าหมายทั้งตำแหน่งและการหมุน
        # =========================================================
        current_q = np.array(node.current_joint_positions)[:6]
        current_slider = node.current_slider_position
        T_cur_list = forward_kinematics_matrices(current_q, L1, L2, L3, D6, current_slider)
        
        R_start = T_cur_list[-1][:3, :3]
        P_start = T_cur_list[-1][:3, 3] 

        initial_Z_dir = R_start[:, 2] 
        
        # 1.1 คำนวณจุดหมายปลายทาง (Position Target)
        P_target = P_start + (initial_Z_dir * distance)

        # 1.2 คำนวณองศาเป้าหมาย (Orientation Target) บังคับ Z=0 ให้ขนานพื้น
        z_tar = np.array([initial_Z_dir[0], initial_Z_dir[1], 0.0])
        Z_target = z_tar / np.linalg.norm(z_tar)
        stuck_counter = 0
        prev_P_curr = P_start.copy()

        # =========================================================
        # 2. ลูปการเคลื่อนที่แบบควบคู่ (Simultaneous Control Loop)
        # =========================================================
        while rclpy.ok():
            current_q = np.array(node.current_joint_positions)[:6]
            T_cur_list = forward_kinematics_matrices(current_q, L1, L2, L3, D6, node.current_slider_position)
            R_curr = T_cur_list[-1][:3, :3]
            P_curr = T_cur_list[-1][:3, 3]

            # --- คำนวณความเร็วเชิงเส้น (Linear Velocity) ---
            P_error = P_target - P_curr
            dist_to_target = np.linalg.norm(P_error)
            
            is_hitting_limit = False
            for i in range(6):
                # เผื่อระยะเซฟตี้ไว้ 0.02 Radian (ประมาณ 1.14 องศา) ก่อนชน Limit จริง
                if JOINT_LIMITS_ENABLED and (current_q[i] <= Q_MIN[i] + 0.02 or current_q[i] >= Q_MAX[i] - 0.02):
                    is_hitting_limit = True
                    break

            if is_hitting_limit:
                print("🛑 ถึงขีดจำกัดข้อต่อ (Joint Limit)! ไปต่อไม่ได้แล้ว หยุดการทำงาน")
                node.publish_joints_velo([0.0]*6, 0.0)
                break

            # =================================================================
            # 🛡️ ระบบป้องกัน 2: ตรวจสอบว่า "ยืดสุดแขน (Singularity/Workspace Limit)" หรือไม่
            # =================================================================
            # หาระยะทางที่ปลายแขนขยับได้ใน 1 ลูป
            step_move = np.linalg.norm(P_curr - prev_P_curr)
            
            # ถ้าขยับได้น้อยมากๆ (น้อยกว่า 0.05 มม. ต่อลูป) แปลว่าสมการเริ่มตัน
            if step_move < 0.00005: 
                stuck_counter += 1
            else:
                stuck_counter = 0

            # ถ้าติดแหง็กแบบนี้ติดต่อกัน 20 ลูป (0.2 วินาที) และยังไม่ถึงเป้าหมาย
            if stuck_counter > 20 and dist_to_target > 0.01:
                print("🛑 หุ่นยนต์ยืดแขนไปจนสุดระยะเอื้อมแล้ว! หยุดการทำงานตรงนี้")
                node.publish_joints_velo([0.0]*6, 0.0)
                break

            prev_P_curr = P_curr.copy()

            Kp_linear = 1.5
            V_linear = Kp_linear * P_error
            if np.linalg.norm(V_linear) > max_linear_speed:
                V_linear = (V_linear / np.linalg.norm(V_linear)) * max_linear_speed

            # --- คำนวณความเร็วเชิงมุม (Angular Velocity) ---
            Z_curr = R_curr[:, 2]
            omega_align = np.cross(Z_curr, Z_target)
            align_error = np.linalg.norm(omega_align)

            Kp_angular = 1.5
            V_angular = Kp_angular * omega_align
            if np.linalg.norm(V_angular) > max_angular_speed:
                V_angular = (V_angular / np.linalg.norm(V_angular)) * max_angular_speed

            # --- ตรวจสอบเงื่อนไขการจบงาน (ต้องผ่านทั้ง 2 เงื่อนไข) ---
            # ระยะห่าง < 5 มม. และ มุมคลาดเคลื่อนน้อยมากๆ
            if dist_to_target < 0.005 and align_error < 0.01:
                print("✅ ถึงเป้าหมายและขนานระนาบ XY เรียบร้อยแล้ว!")
                node.publish_joints_velo([0.0]*6, 0.0)
                break

            # --- นำคำสั่งรวมกันเข้า Jacobian ---
            V_target_base = np.concatenate((V_linear, V_angular))
            J = node.kinematics.get_jacobian(current_q)
            lambda_sq = 0.01
            J_pinv = J.T @ np.linalg.inv(J @ J.T + lambda_sq * np.eye(6))
            q_dot = J_pinv @ V_target_base
            
            q_dot = np.clip(q_dot, -1.0, 1.0)
            node.publish_joints_velo(q_dot.tolist(), 0.0)
            
def run_pose(task):
    target_task = None
    try:
        task_list = json.loads(task)
    except Exception as e:
        print(f"⚠️ แปลงข้อมูล JSON ไม่สำเร็จ: {e}")
        return
        
    if not task_list:
        print("⚠️ โปรแกรมนี้ยังไม่มีข้อมูล Task")
        return
    if isinstance(task_list, dict):
        task_list = [task_list]
    print(task_list)
    for i in task_list:
        pose_cmd = []
        label = i.get('label', 'N/A')
        print(f"\n--- seq: {i.get('sequence', 'N/A')} name: {label} ---")

        for j in range(1, 7):
            val = i.get(f"joint_{j}", 0.0)
            pose_cmd.append(val)
            print(f"moving q{j} to {val}")

        rail = i.get('rail', 0.0)
        speed = i.get('speed', 100)
        delay = i.get('delay')
        gripper = i.get('gripper', 0)
        active_control_mode = i.get('control_mode', i.get('controlMode'))
        pose_cmd.extend([rail, speed, gripper])
        print(f"moving rail to {rail} | speed {speed}% | gripper {gripper} with delay {delay} in {active_control_mode}")
        
        print(f"Move in {active_control_mode}")
        print(i)
        if label == "jog":
            if active_control_mode == 'effector' and node.current_machine_state != 1:
                instant_jog_task(i)
            elif active_control_mode == 'joint' and node.current_machine_state != 1:
                instant_jog_joint(i)        
            else:
                print("Busy or Unknown Control mode")
        elif label =="forward":
            if node.current_machine_state != 1:
                move_forward_and_align(i)
            else:
                print("Robot is busy (working state)")
        else: 
            if active_control_mode == 'effector' and node.current_machine_state != 1:
                move_save_ee(pose_cmd)
            elif active_control_mode == 'joint' and node.current_machine_state != 1:
                move_save(pose_cmd)        
            else:
                print(f"{active_control_mode} ,{node.current_machine_state}")
                print("Busy or Unknown Control mode")


def main(args=None):
    global node
    global jacobian

    rclpy.init(args=args)

    node = JointPublisher()

    ros_thread = threading.Thread(
        target=rclpy.spin,
        args=(node,),
        daemon=True
    )
    ros_thread.start()

    jacobian = RobotVelocityKinematics()

    print("✅ Node หุ่นยนต์ทำงานแล้ว (พร้อม Limit Control)")
    print("📡 รอรับคำสั่งผ่าน Topic: /goto_position และ /force_sensor")

    def _run_xyz_gui():
        BG, PANEL, ACCENT, TEXT, RED, GREEN = '#0f1117','#1a1d27','#00e5ff','#e8eaf6','#ff5252','#69ff6e'
        MUTED, PINK = '#555870', '#ff4081'

        root = tk.Tk()
        root.title('XYZ → /goto_position')
        root.configure(bg=BG)
        root.resizable(False, False)

        fields = [
            ('X (m)',      -0.8,  0.8,  0.0),
            ('Y (m)',      -0.8,  0.8,  0.0),
            ('Z (m)',       0.0,  1.2,  0.5),
            ('Roll (°)',  -180,  180,   0.0),
            ('Pitch (°)', -180,  180,   0.0),
            ('Yaw (°)',   -180,  180,   0.0),
        ]
        vars_ = []

        tk.Label(root, text='XYZ GOTO', font=('Courier New', 12, 'bold'),
                 fg=ACCENT, bg=BG).grid(row=0, column=0, columnspan=3, pady=(12,4), padx=16, sticky='w')
        tk.Frame(root, bg=ACCENT, height=1).grid(row=1, column=0, columnspan=3, sticky='ew', padx=16, pady=(0,8))

        for i, (label, lo, hi, default) in enumerate(fields):
            tk.Label(root, text=label, width=11, anchor='w',
                     font=('Courier New', 9, 'bold'), fg=TEXT, bg=BG).grid(
                         row=i+2, column=0, padx=(16,4), pady=3, sticky='w')
            var = tk.DoubleVar(value=default)
            vars_.append(var)
            ent = tk.Entry(root, textvariable=var, width=10,
                           font=('Courier New', 10), bg=PANEL, fg=ACCENT,
                           insertbackground=ACCENT, relief='flat', bd=4)
            ent.grid(row=i+2, column=1, padx=4, pady=3)

        tk.Frame(root, bg=MUTED, height=1).grid(row=8, column=0, columnspan=3, sticky='ew', padx=16, pady=(8,4))

        # Speed
        tk.Label(root, text='Speed %', width=11, anchor='w',
                 font=('Courier New', 9, 'bold'), fg=TEXT, bg=BG).grid(
                     row=9, column=0, padx=(16,4), pady=3, sticky='w')
        speed_var = tk.IntVar(value=50)
        tk.Entry(root, textvariable=speed_var, width=10,
                 font=('Courier New', 10), bg=PANEL, fg=PINK,
                 insertbackground=PINK, relief='flat', bd=4).grid(row=9, column=1, padx=4, pady=3)

        # Mode
        tk.Label(root, text='Mode', width=11, anchor='w',
                 font=('Courier New', 9, 'bold'), fg=TEXT, bg=BG).grid(
                     row=10, column=0, padx=(16,4), pady=3, sticky='w')
        mode_var = tk.StringVar(value='effector')
        mode_frame = tk.Frame(root, bg=BG)
        mode_frame.grid(row=10, column=1, pady=3, sticky='w')
        for val, lbl in [('effector', 'Effector'), ('joint', 'Joint')]:
            tk.Radiobutton(mode_frame, text=lbl, variable=mode_var, value=val,
                           font=('Courier New', 9), fg=TEXT, bg=BG,
                           selectcolor=PANEL, activebackground=BG,
                           activeforeground=ACCENT).pack(side='left', padx=4)

        # Status label
        status_var = tk.StringVar(value='')
        status_lbl = tk.Label(root, textvariable=status_var,
                              font=('Courier New', 9), fg=RED, bg=BG, wraplength=260)
        status_lbl.grid(row=11, column=0, columnspan=3, padx=16, pady=(4,0))

        def _send():
            try:
                x     = vars_[0].get()
                y     = vars_[1].get()
                z     = vars_[2].get()
                roll  = np.radians(vars_[3].get())
                pitch = np.radians(vars_[4].get())
                yaw   = np.radians(vars_[5].get())
                speed = int(speed_var.get())
                mode  = mode_var.get()
                R_tar = rpy_to_matrix(roll, pitch, yaw)
                q, reachable = inverse_kinematics_6dof(
                    np.array([x, y, z]), R_tar, L1, L2, L3, D6)
                if not reachable:
                    status_var.set('Out of reach')
                    send_btn.config(bg=RED)
                    return
                rail_mm = node.current_slider_position * 1000.0
                node.publish_goto(np.degrees(q).tolist(), rail_mm, speed, mode)
                status_var.set(f'Sent  J={np.degrees(q).round(1).tolist()}')
                send_btn.config(bg=ACCENT)
            except Exception as e:
                status_var.set(f'Error: {e}')

        send_btn = tk.Button(root, text='SEND TO ROBOT',
                             font=('Courier New', 10, 'bold'),
                             bg=ACCENT, fg=BG, activebackground='#00bcd4',
                             bd=0, padx=20, pady=8, cursor='hand2',
                             command=_send)
        send_btn.grid(row=12, column=0, columnspan=3, pady=12, padx=16, sticky='ew')

        root.mainloop()

    threading.Thread(target=_run_xyz_gui, daemon=True).start()

    try:
        while rclpy.ok():
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("กำลังปิดโปรแกรม...")

    finally:
        node.publish_joints_velo([0.0] * 6)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()