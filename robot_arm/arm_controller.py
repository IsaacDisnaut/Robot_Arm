import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String,Int8
from geometry_msgs.msg import WrenchStamped,Pose, Point, Quaternion
import threading
import json
import time

L1, L2, L3 = 0.28787, 0.26096, 0.26136
D6 = 0.07074
J4_OFFSET_Y = 0.02175
A1=0.020885

Q_MIN = np.radians([-180.0, -180.0, -120.0, -180.0, -100.0, -360.0])
Q_MAX = np.radians([ 180.0,  30.0,  150.0,  180.0,  100.0,  360.0])

# กำหนดขีดจำกัดของรางสไลด์ (หน่วยเป็นเมตร)
SLIDER_MIN = -1.0
SLIDER_MAX = 1.0
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
        self.publisher = self.create_publisher(JointState, '/motor', 10)
        self.subscription = self.create_subscription(JointState, '/joint_states', self.joint_cb, 10)
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
        self.current_machine_state= 0
        self.current_joint_positions = [0.0] * 6
        self.current_slider_position = 0.0
        self.has_received_data = False

        self.joints_to_publish =[0.0, -0.174533, 0.261799, 0.0, 0.0, 0.0]
        self.rail_to_publish = 0.0
        self.velocity_to_publish = [0.0] * 7 
        self.timer = self.create_timer(0.02, self.timer_callback)
        
        self.kinematics = RobotVelocityKinematics()
        self.is_force_moving = False
        self.is_busy = False
        self.stop_event = threading.Event()

    def taskcb(self, msg: String):
        if self.current_machine_state == 1:
            print("⚠️ มองข้ามคำสั่งใหม่: หุ่นยนต์กำลังทำงาน (working) อยู่")
            return
        self.task = msg.data
        threading.Thread(target=run_pose, args=(self.task,), daemon=True).start()
   
    def publish_joints(self, joints, base_y):
        # [ADDED JOINT LIMITS] บังคับตำแหน่งที่รับเข้ามาไม่ให้เกินขีดจำกัด
        safe_joints = np.clip(joints, Q_MIN, Q_MAX)
        safe_base_y = np.clip(base_y, SLIDER_MIN, SLIDER_MAX)
        self.joints_to_publish = [float(q) for q in safe_joints]
        self.rail_to_publish = float(safe_base_y)

    def publish_joints_velo(self, joints_vel, slider_vel=0.0):
        # [ADDED JOINT LIMITS] ระบบเบรกอัตโนมัติ เช็คว่าในอนาคตอันใกล้ (0.02s) จะล้ำเส้นไหม
        dt = 0.02
        safe_slider_vel = slider_vel
        next_slider = self.rail_to_publish + slider_vel * dt
        if (next_slider <= SLIDER_MIN and slider_vel < 0) or (next_slider >= SLIDER_MAX and slider_vel > 0):
            safe_slider_vel = 0.0 # ถ้ากำลังจะทะลุลิมิต ให้เบรกเป็น 0 ทันที

        safe_joints_vel = list(joints_vel)
        for i in range(6):
            next_q = self.joints_to_publish[i] + joints_vel[i] * dt
            if (next_q <= Q_MIN[i] and joints_vel[i] < 0) or (next_q >= Q_MAX[i] and joints_vel[i] > 0):
                safe_joints_vel[i] = 0.0 # ถ้ากำลังจะทะลุลิมิต ให้เบรกแกนนั้นเป็น 0 ทันที

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
        
        V_target_linear_ee = force_vector_ee[0:3] * K_linear
        V_target_angular_ee = force_vector_ee[3:6] * K_angular

        current_q = np.array(self.current_joint_positions)[:6]
        current_slider = self.current_slider_position

        T_cur_list = forward_kinematics_matrices(current_q, L1, L2, L3, D6, current_slider)
        T_ee = T_cur_list[-1]
        R_ee = T_ee[:3, :3] 

        V_target_linear_base = R_ee @ V_target_linear_ee
        V_target_angular_base = R_ee @ V_target_angular_ee
        V_target_base = np.concatenate((V_target_linear_base, V_target_angular_base))
        J = self.kinematics.get_jacobian(current_q)
        lambda_sq = 0.01
        J_pinv = J.T @ np.linalg.inv(J @ J.T + lambda_sq * np.eye(6))
        q_dot = J_pinv @ V_target_base
        max_q_dot = 1.0 
        q_dot = np.clip(q_dot, -max_q_dot, max_q_dot)
        self.publish_joints_velo(q_dot.tolist(), 0.0)

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
            # [ADDED JOINT LIMITS] ห้ามรางสไลด์หลุดขอบ (ชั้นสุดท้าย)
            self.rail_to_publish = np.clip(self.rail_to_publish, SLIDER_MIN, SLIDER_MAX)

            for i in range(6):
                self.joints_to_publish[i] += self.velocity_to_publish[i+1] * dt 
                # [ADDED JOINT LIMITS] ห้าม Joint หลุดขอบ (ชั้นสุดท้าย)
                self.joints_to_publish[i] = np.clip(self.joints_to_publish[i], Q_MIN[i], Q_MAX[i])

        msg.position = [float(self.rail_to_publish)] + [float(q) for q in self.joints_to_publish]
        msg.velocity = [float(v) for v in self.velocity_to_publish]
        if self.safety_status ==2 or self.robot_status == 2:
            print("Pause")
        else:
            self.publisher.publish(msg)

# ================= Math & Kinematics =================
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
    dh_params = [
        [q1,           l1, 0,  -np.pi/2], 
        [q2,           0,  l2,  0      ], 
        [q3 + np.pi/2, 0,  0,   np.pi/2], 
        [q4,           l3, 0,  -np.pi/2], 
        [q5,           0,  0,   np.pi/2], 
        [q6,           d6, 0,   0      ]  
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
    # [ADDED JOINT LIMITS] ถ้าตำแหน่ง Home เกินขอบเขต ให้ตัดขอบก่อน
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
        time.sleep(dt) 
        
    node.publish_joints_velo([0.0]*6,0)

def move_save(task):
    # [ADDED LIMIT] ตัดขอบเป้าหมายให้อยู่ในระยะที่ปลอดภัยตั้งแต่ต้น
    tar_q = np.clip(np.radians(task[:6]), Q_MIN, Q_MAX)  
    target_slider = np.clip(task[6]/1000, SLIDER_MIN, SLIDER_MAX)   
    
    speed_pct = task[7] 
    gripper = task[8]        
    node.machine_state(1)
    Kp = 5.0 * (speed_pct / 100.0)
    Kd = 0.1
    dt = 0.05 
    
    prev_error_q = np.zeros(6)
    prev_error_slider = 0.0  
    
    print(f"🚀 [MOVE] เคลื่อนที่ด้วย PD Control (Kp: {Kp}, Kd: {Kd})")

    while True:
        current_q = np.array(node.current_joint_positions) 
        current_slider = node.current_slider_position 

        current_T_list = forward_kinematics_matrices(current_q, L1, L2, L3, D6, current_slider)
        current_T = current_T_list[-1]
        current_pose = matrix_to_rpy(current_T)
        node.ee_pose(current_pose)

        error_q = tar_q - current_q
        derivative_q = (error_q - prev_error_q) / dt
        q_dot = (Kp * error_q) + (Kd * derivative_q)
        
        max_limit = 1.5 
        q_dot = np.clip(q_dot, -max_limit, max_limit)
        reached_mask = np.abs(error_q) < 0.005
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
                time.sleep(0.1)

        node.publish_joints_velo(q_dot.tolist(), float(slider_dot))
        
        prev_error_q = error_q
        prev_error_slider = error_slider
        
        if np.all(reached_mask) and slider_reached:
            print("🎯 [MOVE] ทุกแกนและรางสไลด์เข้าสู่ตำแหน่งเป้าหมายแล้ว!")
            node.machine_state(2)
            break
            
        time.sleep(dt) 
        
    node.publish_joints_velo([0.0]*6, 0.0)

def move_save_ee(task):
    # [ADDED LIMIT] ตัดขอบเป้าหมายปลายทาง
    tar_q = np.clip(np.radians(task[:6]), Q_MIN, Q_MAX)               
    target_slider = np.clip(task[6] / 1000.0, SLIDER_MIN, SLIDER_MAX)   
    speed_pct = task[7] 
    gripper = task[8]
    node.machine_state(1)
    
    Kp_pos = 1.0 * (speed_pct / 100.0)
    Kp_ori = 0.8 * (speed_pct / 100.0)
    Kp_slider = 1.0 * (speed_pct / 100.0) 
    dt = 0.05 
    r_avoid = 0.25 
    z_max = 0.45   
    
    T_tar_list = forward_kinematics_matrices(tar_q, L1, L2, L3, D6, target_slider)
    T_tar = T_tar_list[-1]
    final_tar_p = T_tar[:3, 3]       
    tar_R = T_tar[:3, :3]      
    
    current_q = np.array(node.current_joint_positions)[:6] 
    current_slider = node.current_slider_position
    T_cur_list = forward_kinematics_matrices(current_q, L1, L2, L3, D6, current_slider)
    cur_p = T_cur_list[-1][:3, 3]

    print(f"[PLANNING] เป้าหมายสุดท้าย: {np.round(final_tar_p, 3)}")
    tar_x, tar_y, tar_z = final_tar_p
    tar_radius = np.sqrt(tar_x**2 + tar_y**2)

    if tar_radius < r_avoid and tar_z <= z_max:
        print(f"[ERROR] ยกเลิกคำสั่ง! ตำแหน่งเป้าหมายจมอยู่ในฐานหุ่นยนต์")
        node.machine_state(0)
        node.publish_joints_velo([0.0]*6, 0.0) 
        return
        
    waypoints = []
    p1 = cur_p[:2]
    p2 = final_tar_p[:2]
    v_path = p2 - p1
    path_len = np.linalg.norm(v_path)
    
    if path_len > 0.001:
        v_dir = v_path / path_len
        t = np.dot(-p1, v_dir)
        t = np.clip(t, 0, path_len) 
        closest_p_2d = p1 + t * v_dir
        dist_to_origin = np.linalg.norm(closest_p_2d)

        z_closest = cur_p[2] + (t / path_len) * (final_tar_p[2] - cur_p[2])

        if dist_to_origin < r_avoid and z_closest < z_max:
            print("🚧 [PLANNING] ตรวจพบเส้นทางตัดผ่านฐาน! กำลังสร้างจุดอ้อม...")
            if dist_to_origin < 1e-4: 
                n_vec = np.array([-v_dir[1], v_dir[0]]) 
            else:
                n_vec = closest_p_2d / dist_to_origin
            safe_margin = 0.03
            via_p_2d = n_vec * (r_avoid + safe_margin)
            via_p = np.array([via_p_2d[0], via_p_2d[1], z_closest])
            waypoints.append(via_p)
            print(f"📍 [PLANNING] เพิ่มจุดแวะพัก (Via-point) ที่: {np.round(via_p, 3)}")
            
    waypoints.append(final_tar_p)
    
    current_wp_idx = 0
    prev_q_dot = np.zeros(6)
    loop_count = 0
    aligning_joints = False 

    while True:
        loop_count += 1

        if node.safety_status == 1:
            speed_pct = 0.5 * task[7]
        else:
            speed_pct = task[7]
            
        if node.safety_status == 2 or node.robot_status == 2:
            while node.safety_status == 2 or node.robot_status == 2:
                print("Pause Temporary")
                time.sleep(0.1)
                
        current_q = np.array(node.current_joint_positions)[:6] 
        current_slider = node.current_slider_position
        T_cur_list = forward_kinematics_matrices(current_q, L1, L2, L3, D6, current_slider)
        T_cur = T_cur_list[-1]
        cur_p = T_cur[:3, 3]
        cur_R = T_cur[:3, :3]

        current_pose = matrix_to_rpy(T_cur)
        node.ee_pose(current_pose)

        tar_p = waypoints[current_wp_idx]
        e_pos = tar_p - cur_p
        e_ori = 0.5 * (np.cross(cur_R[:, 0], tar_R[:, 0]) + 
                       np.cross(cur_R[:, 1], tar_R[:, 1]) + 
                       np.cross(cur_R[:, 2], tar_R[:, 2]))
        
        v_norm = np.linalg.norm(e_pos)
        w_norm = np.linalg.norm(e_ori)

        e_joint = tar_q - current_q
        joint_err_max = np.max(np.abs(e_joint))
        e_slider = target_slider - current_slider

        if current_wp_idx < len(waypoints) - 1:
            if v_norm < 0.05:
                print(f"✅ [PLANNING] ถึงจุดแวะพักที่ {current_wp_idx + 1} แล้ว มุ่งหน้าสู่เป้าหมายต่อไป!")
                current_wp_idx += 1
                continue 
        else:
            if v_norm < 0.02 and w_norm < 0.05:
                if not aligning_joints:
                    print("🔄 [PLANNING] เข้าใกล้เป้าหมาย สลับเข้าโหมดจัดองศาข้อต่อ (Joint Alignment)")
                aligning_joints = True

        if aligning_joints:
            Kp_joint = 2.0 * (speed_pct / 100.0)
            q_dot_raw = Kp_joint * e_joint
            
            slider_dot = Kp_slider * e_slider
            slider_dot = np.clip(slider_dot, -0.1, 0.1)
            
            alpha = 0.2 
            q_dot = (alpha * q_dot_raw) + ((1.0 - alpha) * prev_q_dot)
            prev_q_dot = q_dot.copy()
            q_dot = np.clip(q_dot, -0.5, 0.5) 
            
            if joint_err_max < 0.001 and abs(e_slider) < 0.005:
                print("🎯 [COMPLETE] เข้าสู่ตำแหน่งและองศาข้อต่อเป้าหมายสำเร็จ 100%!")
                node.machine_state(2)
                break
                
        else:
            V_pos_target = Kp_pos * e_pos
            max_v = 10 * (speed_pct / 100.0) 
            if v_norm > 0 and np.linalg.norm(V_pos_target) > max_v:
                V_pos_target = V_pos_target * (max_v / np.linalg.norm(V_pos_target))
            
            V_ori_target = Kp_ori * e_ori
            max_w = 0.5 * (speed_pct / 100.0)  
            if w_norm > 0 and np.linalg.norm(V_ori_target) > max_w:
                V_ori_target = V_ori_target * (max_w / np.linalg.norm(V_ori_target))

            V_target_abs = np.concatenate((V_pos_target, V_ori_target))
            
            slider_dot = Kp_slider * e_slider
            slider_dot = np.clip(slider_dot, -0.2, 0.2)
            
            V_base = np.array([0.0, slider_dot, 0.0, 0.0, 0.0, 0.0])
            V_arm_target = V_target_abs - V_base

            J = jacobian.get_jacobian(current_q)
            det_JJT = np.linalg.det(J @ J.T)
            w = np.sqrt(max(0.0, det_JJT))

            w_threshold = 0.05 
            if w < w_threshold:
                lambda_sq = 0.01 * (1.0 - (w / w_threshold)**2)
            else:
                lambda_sq = 0.0 
                
            J_pinv = J.T @ np.linalg.inv(J @ J.T + lambda_sq * np.eye(6))
            q_dot_raw = J_pinv @ V_arm_target
            
            alpha = 0.3 
            q_dot = (alpha * q_dot_raw) + ((1.0 - alpha) * prev_q_dot)
            prev_q_dot = q_dot.copy()
            q_dot = np.clip(q_dot, -1.0, 1.0)

        # [ADDED LIMIT] ตรวจสอบความเร็วก่อนสั่งงาน ถ้าแกนไหนกำลังจะทะลุลิมิต ให้เบรกแกนนั้น
        for i in range(6):
            if (current_q[i] <= Q_MIN[i] and q_dot[i] < 0) or (current_q[i] >= Q_MAX[i] and q_dot[i] > 0):
                q_dot[i] = 0.0
                
        if (current_slider <= SLIDER_MIN and slider_dot < 0) or (current_slider >= SLIDER_MAX and slider_dot > 0):
            slider_dot = 0.0

        if not aligning_joints and np.linalg.norm(q_dot) < 0.005 and abs(slider_dot) < 0.005 and loop_count > 100:
            print("⚠️ [WARNING] หุ่นขยับต่อไม่ได้แล้ว (ติด Singularity หรือ Joint Limits)")
            node.machine_state(3)
            break

        node.publish_joints_velo(q_dot.tolist(), float(slider_dot))
        time.sleep(dt)
    
    node.publish_joints_velo([0.0]*6, 0.0)
    
def instant_jog_joint(task):
    if isinstance(task, list):
        if len(task) > 0:
            task = task[0]
        else:
            return
    axis = task.get('axis', '')       
    direction = task.get('direction', 0) 
    speed = task.get('speed', 0)     

    if direction == 0 or speed == 0:
        return

    max_step_degrees = 1
    step_deg = direction * (speed / 100.0) * max_step_degrees
    step_rad = np.radians(step_deg)
    max_step_slider_mm = 5.0 
    step_slider_m = direction * (speed / 100.0) * max_step_slider_mm / 1000.0

    step_q = np.zeros(6)
    step_slider = 0.0
    if node.safety_status ==2 or node.robot_status ==2:
        while node.safety_status ==2 or node.robot_status ==2:
            print("Pause Temporary")
            time.sleep(0.1)

    if axis == 'joint_1': step_q[0] = step_rad
    elif axis == 'joint_2': step_q[1] = step_rad
    elif axis == 'joint_3': step_q[2] = step_rad
    elif axis == 'joint_4': step_q[3] = step_rad
    elif axis == 'joint_5': step_q[4] = step_rad
    elif axis == 'joint_6': step_q[5] = step_rad
    elif axis in ['rail', 'slider']: step_slider = step_slider_m

    current_target_q = np.array(node.joints_to_publish)
    current_target_slider = node.rail_to_publish

    new_q = current_target_q + step_q
    new_slider = current_target_slider + step_slider

    # [ADDED LIMIT] ตัดขอบค่าที่คำนวณได้ให้อยู่ในระยะขีดจำกัด
    clamped_q = np.clip(new_q, Q_MIN, Q_MAX)
    clamped_slider = np.clip(new_slider, SLIDER_MIN, SLIDER_MAX)
    print(f"🔍 DEBUG:")
    print(f"  เป้าหมายที่คำนวณได้ (เรเดียน): {np.round(new_q, 3)}")
    print(f"  ค่า Q_MIN (เรเดียน): {np.round(Q_MIN, 3)}")
    print(f"  ค่า Q_MAX (เรเดียน): {np.round(Q_MAX, 3)}")
    print(f"  ค่าที่โดน Clamp แล้ว: {np.round(clamped_q, 3)}")
    if not np.allclose(new_q, clamped_q) or not np.isclose(new_slider, clamped_slider):
         print(f"⚠️ [JOG JOINT WARNING] แกน {axis} ถึงขีดจำกัด (Limit) แล้ว! จะหยุดที่ระยะปลอดภัย")

    node.publish_joints(clamped_q.tolist(), float(clamped_slider))
    
    current_q = np.array(node.current_joint_positions)[:6] 
    current_slider = node.current_slider_position
    T_cur_list = forward_kinematics_matrices(current_q, L1, L2, L3, D6, current_slider)
    T_cur = T_cur_list[-1]
    current_pose = matrix_to_rpy(T_cur)
    node.ee_pose(current_pose)

    print(f"⚡ [JOG JOINT] ขยับแกน {axis} | ทิศ: {direction} | ความเร็ว: {speed}%")

def instant_jog_task(task):
    if isinstance(task, list) and len(task) > 0:
        task = task[0]

    axis = task.get('axis', '')         
    direction = task.get('direction', 0) 
    speed = task.get('speed', 0)        

    tcp_x = task.get('tcp_x', 0.0) / 1000.0
    tcp_y = task.get('tcp_y', 0.0) / 1000.0
    tcp_z = task.get('tcp_z', 0.0) / 1000.0
    tcp_offset = np.array([tcp_x, tcp_y, tcp_z])

    if direction == 0 or speed == 0:
        return

    max_step_m = 5.0 / 1000.0  
    max_step_rad = np.radians(1) 

    step_pos = np.zeros(3)
    step_rpy = np.zeros(3)
    
    if node.safety_status ==2 or node.robot_status ==2:
        while node.safety_status ==2 or node.robot_status ==2:
            print("Pause Temporary")
            time.sleep(0.1)
            
    if axis == 'x': step_pos[0] = direction * (speed / 100.0) * max_step_m
    elif axis == 'y': step_pos[1] = direction * (speed / 100.0) * max_step_m
    elif axis == 'z': step_pos[2] = direction * (speed / 100.0) * max_step_m
    elif axis == 'roll': step_rpy[0] = direction * (speed / 100.0) * max_step_rad
    elif axis == 'pitch': step_rpy[1] = direction * (speed / 100.0) * max_step_rad
    elif axis == 'yaw': step_rpy[2] = direction * (speed / 100.0) * max_step_rad

    current_q = np.array(node.joints_to_publish)
    current_slider = node.rail_to_publish

    T_list = forward_kinematics_matrices(current_q, L1, L2, L3, D6, base_y=0.0)
    T_end_effector = T_list[-1] 
    
    current_flange_pos = T_end_effector[:3, 3]    
    current_flange_orient = T_end_effector[:3, :3]
    
    current_q = np.array(node.current_joint_positions)[:6] 
    current_slider = node.current_slider_position
    T_cur_list = forward_kinematics_matrices(current_q, L1, L2, L3, D6, current_slider)
    T_cur = T_cur_list[-1]
    current_pose = matrix_to_rpy(T_cur)
    node.ee_pose(current_pose)
    
    current_tcp_pos = current_flange_pos + (current_flange_orient @ tcp_offset)
    target_tcp_pos = current_tcp_pos + step_pos
    step_orient = rpy_to_matrix(step_rpy[0], step_rpy[1], step_rpy[2])
    target_tcp_orient = step_orient @ current_flange_orient 
    target_flange_pos = target_tcp_pos - (target_tcp_orient @ tcp_offset)
    
    new_q, reachable = inverse_kinematics_6dof(target_flange_pos, target_tcp_orient, L1, L2, L3, D6)
    
    t_x, t_y, t_z = target_tcp_pos
    in_zone_x = -0.2 <= t_x <= 0.2
    in_zone_y = -0.2 <= t_y <= 0.2
    in_zone_z = 0.0 <= t_z <= 0.4
    
    is_in_restricted_zone = in_zone_x and in_zone_y and in_zone_z
    
    # [ADDED LIMIT] ตรวจสอบว่าพิกัด IK ที่คำนวณได้ ล้ำเส้น Joint Limit หรือไม่
    within_limits = np.all((new_q >= Q_MIN) & (new_q <= Q_MAX))

    if is_in_restricted_zone:
        print(f"⚠️[JOG WARNING] Arm in base area (x:{t_x:.3f}, y:{t_y:.3f}, z:{t_z:.3f})")
    elif reachable and within_limits:
        node.publish_joints(new_q.tolist(), float(current_slider))
        print(f"[JOG GLOBAL] move {axis} success! | in direction: {direction}")
    else:
        if not within_limits:
            print(f"⛔[JOG ERROR] คำสั่ง Jog {axis} ทำให้ข้อต่อหมุนเกิน Joint Limit! (คำสั่งถูกยกเลิก)")
        else:
            print(f"⛔[JOG ERROR] {axis} Out of range หรือเกิด Singularity")
        
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
        else: 
            if active_control_mode == 'effector' and node.current_machine_state != 1:
                move_save_ee(pose_cmd)
            elif active_control_mode == 'joint' and node.current_machine_state != 1:
                move_save(pose_cmd)        
            else:
                print(f"{active_control_mode} ,{node.current_machine_state}")
                print("Busy or Unknown Control mode")

# ================= Main Execution =================
if __name__ == '__main__':
    rclpy.init()
    node = JointPublisher()
    
    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()

    jacobian = RobotVelocityKinematics()

    print("✅ Node หุ่นยนต์ทำงานแล้ว (พร้อม Limit Control)")
    print("📡 รอรับคำสั่งผ่าน Topic: /goto_position และ /force_sensor")
    
    try:
        while rclpy.ok():
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("กำลังปิดโปรแกรม...")
    finally:
        node.publish_joints_velo([0.0]*6)
        node.destroy_node()
        rclpy.shutdown()