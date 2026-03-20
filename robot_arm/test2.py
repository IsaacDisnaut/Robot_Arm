import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from geometry_msgs.msg import WrenchStamped  # 🟢 Import สำหรับรับค่า Force Sensor
import threading
import json
import time

L1, L2, L3 = 0.6, 0.6, 0.6
D6 = 0.1
J4_OFFSET_Y = 0

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
        d6 = 0.07074
        offset_y = 0
        a1 = 0
        
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
        self.publisher = self.create_publisher(JointState, '/joint_states', 10)
        self.subscription = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_cb,
            10)
        
        self.tasksub = self.create_subscription(
            String,
            '/goto_position',
            self.taskcb,
            10
        )
        
        # 🟢 เพิ่ม Subscription รับค่าจาก Force Sensor
        self.force_sub = self.create_subscription(
            WrenchStamped,
            '/force_sensor',
            self.force_cb,
            10
        )
        
        self.machine_state_pub = self.create_publisher(String, '/machine_state', 10)
        self.current_machine_state= "Idle"
        self.current_joint_positions = [0.0] * 6
        self.current_slider_position = 0.0
        self.has_received_data = False

        self.joints_to_publish = [0.0] * 6
        self.rail_to_publish = 0.0
        self.velocity_to_publish = [0.0] * 7 
        self.timer = self.create_timer(0.02, self.timer_callback)
        
        # 🟢 สร้าง Object ของ Kinematics ไว้ใช้คำนวณแรง
        self.kinematics = RobotVelocityKinematics()
        self.is_force_moving = False
        self.is_busy = False
        self.stop_event = threading.Event()

    def taskcb(self, msg: String):
        if self.current_machine_state == "working":
            print("⚠️ มองข้ามคำสั่งใหม่: หุ่นยนต์กำลังทำงาน (working) อยู่")
            return
        self.task = msg.data
        threading.Thread(target=run_pose, args=(self.task,), daemon=True).start()
   
    def publish_joints(self, joints, base_y):
        self.joints_to_publish = [float(q) for q in joints]
        self.rail_to_publish = float(base_y)

    def publish_joints_velo(self, joints, slider=0.0):
        self.velocity_to_publish = [float(slider)] + [float(q) for q in joints]

    def machine_state(self, stat):
        out_msg = String()
        self.current_machine_state = stat
        out_msg.data = stat
        self.machine_state_pub.publish(out_msg)

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

        # ถ้าแรงน้อยเกินไป (ป้องกัน Noise) ให้สั่งหุ่นหยุด
        if np.linalg.norm(force_vector_ee) < 0.05:
            if self.is_force_moving:
                self.publish_joints_velo([0.0]*6, 0.0)
                self.is_force_moving = False
            return

        self.is_force_moving = True

        K_linear = 0.02  # m/s ต่อ 1 นิวตัน
        K_angular = 0.05 # rad/s ต่อ 1 นิวตัน-เมตร
        
        V_target_linear_ee = force_vector_ee[0:3] * K_linear
        V_target_angular_ee = force_vector_ee[3:6] * K_angular

        # ดึงค่า Joint ปัจจุบัน
        current_q = np.array(self.current_joint_positions)[:6]
        current_slider = self.current_slider_position

        # 🟢 1. คำนวณ Forward Kinematics เพื่อดึง Rotation Matrix ของปลายแขน
        T_cur_list = forward_kinematics_matrices(current_q, L1, L2, L3, D6, current_slider)
        T_ee = T_cur_list[-1]
        R_ee = T_ee[:3, :3] # R_ee คือ Rotation Matrix แปลงจากแกน EE ไปแกน Base

        V_target_linear_base = R_ee @ V_target_linear_ee
        V_target_angular_base = R_ee @ V_target_angular_ee
        V_target_base = np.concatenate((V_target_linear_base, V_target_angular_base))
        J = self.kinematics.get_jacobian(current_q)
        lambda_sq = 0.01
        J_pinv = J.T @ np.linalg.inv(J @ J.T + lambda_sq * np.eye(6))
        q_dot = J_pinv @ V_target_base

        # ลิมิตความเร็วข้อมือ ป้องกันหุ่นสะบัด
        max_q_dot = 1.0 
        q_dot = np.clip(q_dot, -max_q_dot, max_q_dot)
        print(q_dot)
        # สั่งขยับ!
        self.publish_joints_velo(q_dot.tolist(), 0.0)

    def timer_callback(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = ['slider_joint', 'joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']
        
        dt = 0.02
        if any(v != 0.0 for v in self.velocity_to_publish):
            self.rail_to_publish += self.velocity_to_publish[0] * dt
            for i in range(6):
                self.joints_to_publish[i] += self.velocity_to_publish[i+1] * dt 

        msg.position = [float(self.rail_to_publish)] + [float(q) for q in self.joints_to_publish]
        msg.velocity = [float(v) for v in self.velocity_to_publish]
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
    a1 = 0
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

    if np.abs(R36[2, 2]) > 0.9999: 
        q4 = 0
        q6 = np.arctan2(R36[1,0], R36[0,0])
    else:
        q4 = np.arctan2(R36[1,2], R36[0,2])
        q6 = np.arctan2(R36[2,1], -R36[2,0])

    return np.array([q1, q2, q3, q4, q5, q6]), reachable


# ================= Motion Control =================
def reset_home():
    tar_q = [0.0, -1.5708, 1.5708, 0, 0, 0]
    speed_pct = 40       
    Kp = 5.0 * (speed_pct / 100.0)
    Kd = 0.1
    prev_error_q = np.zeros(6)
    dt = 0.05
    node.machine_state("working")
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
    tar_q = np.radians(task[:6])  
    target_slider = task[6]/1000   
    speed_pct = task[7] 
    gripper = task[8]        
    
    node.machine_state("working")
    Kp = 5.0 * (speed_pct / 100.0)
    Kd = 0.1
    dt = 0.05 
    start_q = np.array(node.current_joint_positions)
    start_slider = node.current_slider_position
    start_pose = forward_kinematics_matrices(start_q, L1, L2, L3, D6, base_y=start_slider)[-1]
    start_xyz = start_pose[:3, 3]
    target_pose = forward_kinematics_matrices(tar_q, L1, L2, L3, D6, base_y=target_slider)[-1]
    target_xyz = target_pose[:3, 3]
    target_orient = target_pose[:3, :3] 
    step_size_cartesian = 0.1 
    step_size_slider = 0.01
    
    distance_xyz = np.linalg.norm(target_xyz - start_xyz)
    delta_slider = abs(target_slider - start_slider)
    steps_xyz = int(distance_xyz / step_size_cartesian)
    steps_s = int(delta_slider / step_size_slider)
    num_steps = max(steps_xyz, steps_s, 1) 
    waypoints_xyz = np.linspace(start_xyz, target_xyz, num_steps + 1)[1:]
    waypoints_slider = np.linspace(start_slider, target_slider, num_steps + 1)[1:]
    
    print(f"🚀 [MOVE] เริ่มเคลื่อนที่เชิงเส้นตรง แบ่งการกระจัดออกเป็น {num_steps} จุดย่อย...")
    for i in range(num_steps):
        local_target_pos = waypoints_xyz[i].copy()
        local_target_pos[1] -= waypoints_slider[i] 
        
        ik_q, reachable = inverse_kinematics_6dof(local_target_pos, target_orient, L1, L2, L3, D6)
        sub_tar_q = ik_q if reachable else tar_q 
        sub_tar_slider = waypoints_slider[i]
        
        prev_error_q = np.zeros(6)
        prev_error_slider = 0.0  
        while True:
            current_q = np.array(node.current_joint_positions) 
            current_slider = node.current_slider_position 
            error_q = sub_tar_q - current_q
            derivative_q = (error_q - prev_error_q) / dt
            q_dot = (Kp * error_q) + (Kd * derivative_q)
            
            max_limit = 1.5 
            q_dot = np.clip(q_dot, -max_limit, max_limit)
            reached_mask = np.abs(error_q) < 0.005
            q_dot[reached_mask] = 0.0 
            error_slider = sub_tar_slider - current_slider
            derivative_slider = (error_slider - prev_error_slider) / dt
            slider_dot = (Kp * error_slider) + (Kd * derivative_slider)
            
            max_slider_limit = 0.5 
            slider_dot = np.clip(slider_dot, -max_slider_limit, max_slider_limit)

            slider_reached = abs(error_slider) < 0.005
            if slider_reached:
                slider_dot = 0.0 
            node.publish_joints_velo(q_dot.tolist(), float(slider_dot))
            
            prev_error_q = error_q
            prev_error_slider = error_slider
            if np.all(reached_mask) and slider_reached:
                break
            
    # เมื่อจบการวนลูป Waypoints ทั้งหมด
    node.publish_joints_velo([0.0]*6, 0.0)
    print("🎯 [MOVE] ทุกแกนและรางสไลด์เข้าสู่เป้าหมายสุดท้ายแล้ว!")
    node.machine_state("task_complete")

def move_save_ee(task):
    tar_q = np.radians(task[:6])               
    target_slider = task[6] / 1000.0   
    speed_pct = task[7] 
    gripper = task[8]
    node.machine_state("working")
    Kp_pos = 1.0 * (speed_pct / 100.0)
    Kp_ori = 0.8 * (speed_pct / 100.0)
    Kp_slider = 1.0 * (speed_pct / 100.0) 
    dt = 0.05 
    
    T_tar_list = forward_kinematics_matrices(tar_q, L1, L2, L3, D6, target_slider)
    T_tar = T_tar_list[-1]
    tar_p = T_tar[:3, 3]       
    tar_R = T_tar[:3, :3]      

    print(f"🚀 [JACOBIAN_MOVE] เดินทางเส้นตรงไปที่ XYZ: {np.round(tar_p, 3)}")
    prev_q_dot = np.zeros(6)
    max_iterations = 600
    loop_count = 0

    while True:
        loop_count += 1
        
        current_q = np.array(node.current_joint_positions)[:6] 
        current_slider = node.current_slider_position
        
        T_cur_list = forward_kinematics_matrices(current_q, L1, L2, L3, D6, current_slider)
        T_cur = T_cur_list[-1]
        cur_p = T_cur[:3, 3]
        cur_R = T_cur[:3, :3]

        e_pos = tar_p - cur_p
        e_ori = 0.5 * (np.cross(cur_R[:, 0], tar_R[:, 0]) + 
                       np.cross(cur_R[:, 1], tar_R[:, 1]) + 
                       np.cross(cur_R[:, 2], tar_R[:, 2]))
        
        v_norm = np.linalg.norm(e_pos)
        w_norm = np.linalg.norm(e_ori)
        V_pos_target = Kp_pos * e_pos
        max_v = 10 * (speed_pct / 100.0) 
        if v_norm > 0 and np.linalg.norm(V_pos_target) > max_v:
            V_pos_target = V_pos_target * (max_v / np.linalg.norm(V_pos_target))

        V_ori_target = Kp_ori * e_ori
        max_w = 0.5 * (speed_pct / 100.0)  
        if w_norm > 0 and np.linalg.norm(V_ori_target) > max_w:
            V_ori_target = V_ori_target * (max_w / np.linalg.norm(V_ori_target))

        V_target_abs = np.concatenate((V_pos_target, V_ori_target))

        e_slider = target_slider - current_slider
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
        ee_err = np.max(np.abs(e_pos))
        
        print(f"Velocity: {v_norm:.4f} ,Angular: {w_norm:.4f} ,End effector error: {ee_err:.4f}")
        if ((v_norm < 0.005 and w_norm < 0.05) or ee_err < 0.001) and abs(e_slider) < 0.005:
            print("🎯 [JACOBIAN_MOVE] เข้าสู่ตำแหน่งเป้าหมายแล้ว!")
            node.machine_state("task_complete")
            break
            
        if np.linalg.norm(q_dot) < 0.01 and abs(slider_dot) < 0.005 and loop_count > 50:
            print("⚠️ [WARNING] หุ่นขยับต่อไม่ได้แล้ว (ติด Singularity หรือ Joint Limits) ขอจบคำสั่งนี้")
            node.machine_state("singularity")
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

    if axis == 'j1':
        step_q[0] = step_rad
    elif axis == 'j2':
        step_q[1] = step_rad
    elif axis == 'j3':
        step_q[2] = step_rad
    elif axis == 'j4':
        step_q[3] = step_rad
    elif axis == 'j5':
        step_q[4] = step_rad
    elif axis == 'j6':
        step_q[5] = step_rad
    elif axis in ['rail', 'slider']: 
        step_slider = step_slider_m

    current_target_q = np.array(node.joints_to_publish)
    current_target_slider = node.rail_to_publish

    new_q = current_target_q + step_q
    new_slider = current_target_slider + step_slider

    node.publish_joints(new_q.tolist(), float(new_slider))
    
    print(f"⚡ [JOG JOINT] ขยับแกน {axis} | ทิศ: {direction} | ความเร็ว: {speed}%")

def instant_jog_task(task):
    # 1. ดักจับ Error เผื่อส่งมาเป็น List
    if isinstance(task, list) and len(task) > 0:
        task = task[0]

    axis = task.get('axis', '')         
    direction = task.get('direction', 0) 
    speed = task.get('speed', 0)        

    # 🟢 รับค่า TCP Offset แปลงเป็นเมตร
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

    # ระยะก้าวในแนวแกนโลก (Global)
    if axis == 'x': step_pos[0] = direction * (speed / 100.0) * max_step_m
    elif axis == 'y': step_pos[1] = direction * (speed / 100.0) * max_step_m
    elif axis == 'z': step_pos[2] = direction * (speed / 100.0) * max_step_m
    elif axis == 'roll': step_rpy[0] = direction * (speed / 100.0) * max_step_rad
    elif axis == 'pitch': step_rpy[1] = direction * (speed / 100.0) * max_step_rad
    elif axis == 'yaw': step_rpy[2] = direction * (speed / 100.0) * max_step_rad

    # 3. ดึงค่าองศาปัจจุบัน
    current_q = np.array(node.joints_to_publish)
    current_slider = node.rail_to_publish

    # 4. ใช้ FK หาพิกัด "หน้าแปลน" (Flange) ปัจจุบัน
    T_list = forward_kinematics_matrices(current_q, L1, L2, L3, D6, base_y=0.0)
    T_end_effector = T_list[-1] 
    
    current_flange_pos = T_end_effector[:3, 3]    
    current_flange_orient = T_end_effector[:3, :3]
    
    # 5. หาพิกัด "ปลายทูล (TCP)" จริงๆ ปัจจุบัน
    current_tcp_pos = current_flange_pos + (current_flange_orient @ tcp_offset)
    
    # ==========================================
    # 🟢 6. ส่วนการขยับแบบ GLOBAL FRAME 100%
    # ==========================================
    # ขยับตำแหน่งปลายทูล ตามแกน X, Y, Z ของโลกตรงๆ
    target_tcp_pos = current_tcp_pos + step_pos
    
    # คำนวณการหมุนตามแกนโลก (Global Rotation)
    step_orient = rpy_to_matrix(step_rpy[0], step_rpy[1], step_rpy[2])
    target_tcp_orient = step_orient @ current_flange_orient 
    # ==========================================
    
    # 7. หักลบ TCP Offset ออกจากทิศทางใหม่ เพื่อหาว่า "หน้าแปลน" ต้องถอยกลับไปอยู่ตรงไหน
    target_flange_pos = target_tcp_pos - (target_tcp_orient @ tcp_offset)
    
    # 8. โยนพิกัด "หน้าแปลน" เป้าหมาย เข้าฟังก์ชัน IK 
    new_q, reachable = inverse_kinematics_6dof(target_flange_pos, target_tcp_orient, L1, L2, L3, D6)
    
    # 9. เช็คความปลอดภัยก่อน Publish
    if reachable:
        node.publish_joints(new_q.tolist(), float(current_slider))
        print(f"⚡ [JOG GLOBAL] ขยับแกน {axis} สำเร็จ! | ทิศ: {direction}")
    else:
        print(f"⚠️ [JOG ERROR] ขยับไม่ได้! แกน {axis} จุดนี้อยู่นอกระยะ หรือติด Singularity")
        
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
            val = i.get(f"j{j}", 0.0)
            pose_cmd.append(val)
            print(f"moving q{j} to {val}")

        rail = i.get('rail', 0.0)
        speed = i.get('speed', 100)
        delay = i.get('delay')
        gripper = i.get('gripper', 0)
        control_mode = i.get('controlMode')
        pose_cmd.extend([rail, speed, gripper])
        print(f"moving rail to {rail} | speed {speed}% | gripper {gripper} with delay {delay} in {control_mode}")
        
        print(f"Move in {control_mode}")
        print(i)
        if label == "jog":
            if control_mode == 'effector' and node.current_machine_state != 'working':
                instant_jog_task(i)
            elif control_mode == 'joint' and node.current_machine_state != 'working':
                instant_jog_joint(i)        
            else:
                print("Busy or Unknown Control mode")
        else: 
            if  node.current_machine_state != 'working':
                move_save_ee(pose_cmd)
            elif control_mode == 'joint' and node.current_machine_state != 'working':
                move_save(pose_cmd)        
            else:
                print(f"{control_mode} ,{node.current_machine_state}")
                print("Busy or Unknown Control mode")

# ================= Main Execution =================
if __name__ == '__main__':
    rclpy.init()
    node = JointPublisher()
    
    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()

    jacobian = RobotVelocityKinematics()

    print("✅ Node หุ่นยนต์ทำงานแล้ว (พร้อม Admittance Control)")
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