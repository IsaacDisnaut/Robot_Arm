import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String
import threading
import json
import time

TASK_FILE_PATH = '/home/isaac/ros2_ws/src/robot_arm/tasks/data.txt'
L1, L2, L3 = 0.28787, 0.26096, 0.26136
D6 = 0.07074
J4_OFFSET_Y = 0.02175

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
        l1, l2, l3 = 0.28787, 0.26096, 0.26136
        d6 = 0.07074
        offset_y = 0.02175 
        a1 = 0.020885
        
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
        self.current_joint_positions = [0.0] * 6
        self.current_slider_position = 0.0
        self.has_received_data = False

        self.joints_to_publish = [0.0] * 6
        self.rail_to_publish = 0.0
        self.velocity_to_publish = [0.0] * 7 
        self.timer = self.create_timer(0.02, self.timer_callback)
    
    def taskcb(self, msg: String):
        self.task = msg.data
        print(self.task)
        threading.Thread(target=run_pose, args=(self.task,), daemon=True).start()
   
    # 🟢 1. แก้ไขฟังก์ชันรับค่า Position
    def publish_joints(self, joints, base_y):
        # แยกเก็บค่าข้อต่อและรางสไลด์
        self.joints_to_publish = [float(q) for q in joints]
        self.rail_to_publish = float(base_y)

    # 🟢 2. แก้ไขฟังก์ชันรับค่า Velocity
    def publish_joints_velo(self, joints, slider=0.0):
        # 🌟 ใช้การ "ต่อ List" (Concatenation) แทนการบวกเลข
        # List ของ slider (1 ตัว) + List ของ joints (6 ตัว) = List ใหม่ 7 ตัว
        self.velocity_to_publish = [float(slider)] + [float(q) for q in joints]

    def joint_cb(self, msg):
        if len(msg.position) >= 7:
            self.current_slider_position = msg.position[0]
            self.current_joint_positions = list(msg.position[1:7])
            self.has_received_data = True
            
            if not any(self.joints_to_publish): 
                self.joints_to_publish = list(msg.position[1:7])
                self.rail_to_publish = msg.position[0]

    def timer_callback(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = ['slider_joint', 'joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']
        
        dt = 0.02
        if any(v != 0.0 for v in self.velocity_to_publish):
            self.rail_to_publish += self.velocity_to_publish[0] * dt
            for i in range(6):
                # ตอนนี้ velocity_to_publish มี 7 ตัวแล้ว บรรทัดนี้จะไม่ Error IndexError แน่นอน
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
    # 🔥 เลื่อนฐานของหุ่นยนต์ไปตามแนวแกน Y ของโลก
    T[1, 3] = base_y 
    
    T_list = [T.copy()]
    for params in dh_params:
        T = T @ get_transform(*params)
        T_list.append(T.copy())
    return T_list

def inverse_kinematics_6dof(local_target_pos, target_orient, l1, l2, l3, d6):
    xc, yc, zc = local_target_pos - d6 * target_orient[:, 2]
    a1 = 0.020885
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
        time.sleep(dt) # 🔥 แทนที่ plt.pause()
        
    node.publish_joints_velo([0.0]*6,0)

def move_save(task):
    tar_q = np.radians(task[:6])  
    target_slider = task[6]/1000   
    speed_pct = task[7] 
    gripper = task[8]        
    
    Kp = 5.0 * (speed_pct / 100.0)
    Kd = 0.1
    dt = 0.05 
    
    # ตัวแปรเก็บ Error รอบก่อนหน้าสำหรับหา Derivative
    prev_error_q = np.zeros(6)
    prev_error_slider = 0.0  # 🟢 เพิ่มของรางสไลด์
    
    print(f"🚀 [MOVE] เคลื่อนที่ด้วย PD Control (Kp: {Kp}, Kd: {Kd})")

    while True:
        # 1. ดึง Feedback ปัจจุบัน
        current_q = np.array(node.current_joint_positions) 
        current_slider = node.current_slider_position # เป็นค่า float ปกติ
    
        error_q = tar_q - current_q
        derivative_q = (error_q - prev_error_q) / dt
        q_dot = (Kp * error_q) + (Kd * derivative_q)
        
        max_limit = 1.5 # rad/s
        q_dot = np.clip(q_dot, -max_limit, max_limit)
        reached_mask = np.abs(error_q) < 0.005
        q_dot[reached_mask] = 0.0 # ตัวไหนถึงแล้วให้หยุด
        
        error_slider = target_slider - current_slider
        derivative_slider = (error_slider - prev_error_slider) / dt
        slider_dot = (Kp * error_slider) + (Kd * derivative_slider)
        
        max_slider_limit = 0.5 
        slider_dot = np.clip(slider_dot, -max_slider_limit, max_slider_limit)

        slider_reached = abs(error_slider) < 0.005
        if slider_reached:
            slider_dot = 0.0 

        node.publish_joints_velo(q_dot.tolist(), float(slider_dot))
        
        # เก็บ Error ไว้ใช้ในรอบถัดไป
        prev_error_q = error_q
        prev_error_slider = error_slider
        
        # 🟢 ตรวจสอบเงื่อนไขการหยุด (ต้องถึงเป้าหมายทั้งแขนและราง)
        if np.all(reached_mask) and slider_reached:
            print("🎯 [MOVE] ทุกแกนและรางสไลด์เข้าสู่ตำแหน่งเป้าหมายแล้ว!")
            break
            
        time.sleep(dt) 
        
    # สั่งให้มอเตอร์และรางหยุดสนิทเมื่อจบการทำงาน
    node.publish_joints_velo([0.0]*6, 0.0)

def move_save_ee(task):
    tar_q = np.radians(task[:6])               
    target_slider = task[6] / 1000.0   
    speed_pct = task[7] 
    gripper = task[8]
    
    # แยก Gain สำหรับแต่ละประเภท
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
    # 🟢 เพิ่มตัวนับรอบ ป้องกัน Infinite Loop
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

        # ==========================================
        # 🟢 แก้ไข 1: แยกจำกัดความเร็ว Position และ Orientation 
        # ==========================================
        V_pos_target = Kp_pos * e_pos
        max_v = 10 * (speed_pct / 100.0) # ลิมิตความเร็วเคลื่อนที่ ม./วิ.
        if v_norm > 0 and np.linalg.norm(V_pos_target) > max_v:
            V_pos_target = V_pos_target * (max_v / np.linalg.norm(V_pos_target))

        V_ori_target = Kp_ori * e_ori
        max_w = 0.5 * (speed_pct / 100.0)  # ลิมิตความเร็วหมุนข้อมือ rad/วิ.
        if w_norm > 0 and np.linalg.norm(V_ori_target) > max_w:
            V_ori_target = V_ori_target * (max_w / np.linalg.norm(V_ori_target))

        V_target_abs = np.concatenate((V_pos_target, V_ori_target))
        
        # จัดการสไลเดอร์
        e_slider = target_slider - current_slider
        slider_dot = Kp_slider * e_slider
        slider_dot = np.clip(slider_dot, -0.2, 0.2)
        
        V_base = np.array([0.0, slider_dot, 0.0, 0.0, 0.0, 0.0])
        V_arm_target = V_target_abs - V_base

        J = jacobian.get_jacobian(current_q)

        det_JJT = np.linalg.det(J @ J.T)
        w = np.sqrt(max(0.0, det_JJT))

        w_threshold = 0.05 # ระยะเริ่มทำงาน
        if w < w_threshold:
            lambda_sq = 0.01 * (1.0 - (w / w_threshold)**2)
        else:
            lambda_sq = 0.0 
            
        # คำนวณ Pseudo-inverse แบบ DLS
        J_pinv = J.T @ np.linalg.inv(J @ J.T + lambda_sq * np.eye(6))
        
        # 3. คำนวณความเร็วดิบ
        q_dot_raw = J_pinv @ V_arm_target
        
        alpha = 0.3 # ความสมูท (0.0 ถึง 1.0) ยิ่งค่าน้อยยิ่งสมูท แต่จะตอบสนองช้าลงนิดนึง
        q_dot = (alpha * q_dot_raw) + ((1.0 - alpha) * prev_q_dot)
        
        # อัปเดตค่าความเร็วรอบนี้เก็บไว้ใช้รอบหน้า
        prev_q_dot = q_dot.copy()
        
        # จำกัดความเร็วมอเตอร์
        q_dot = np.clip(q_dot, -1.0, 1.0)
        ee_err = np.max(np.abs(e_pos))
        # ==========================================
        # 🟢 แก้ไข 3: เงื่อนไขการหยุด (Safety & Tolerance)
        # ==========================================
        print(f"Velocity: {v_norm} ,Angular: {w_norm} ,End effector error: {ee_err}")
        if ((v_norm < 0.005 and w_norm < 0.05) or ee_err < 0.01) and abs(e_slider) < 0.005:
            print("🎯 [JACOBIAN_MOVE] เข้าสู่ตำแหน่งเป้าหมายแล้ว!")
            break
            
        # ถ้าความเร็วที่สั่งแทบจะเป็น 0 แล้ว แต่ error ยังไม่ผ่านเกณฑ์ (เช่น ติด Limit เชิงกล)
        if np.linalg.norm(q_dot) < 0.01 and abs(slider_dot) < 0.005 and loop_count > 50:
            print("⚠️ [WARNING] หุ่นขยับต่อไม่ได้แล้ว (ติด Singularity หรือ Joint Limits) ขอจบคำสั่งนี้")
            break

        node.publish_joints_velo(q_dot.tolist(), float(slider_dot))
        time.sleep(dt)
    
    # สั่งให้มอเตอร์และรางหยุดสนิท
    node.publish_joints_velo([0.0]*6, 0.0)


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

    for i in task_list:
        pose_cmd = []
        print(f"\n--- seq: {i.get('sequence', 'N/A')} name: {i.get('label', 'N/A')} ---")

        for j in range(1, 7):
            val = i.get(f"j{j}", 0.0)
            pose_cmd.append(val)
            print(f"moving q{j} to {val}")

        rail = i.get('rail', 0.0)
        speed = i.get('speed', 100)
        gripper = i.get('gripper', 0)

        pose_cmd.extend([rail, speed, gripper])
        print(f"moving rail to {rail} | speed {speed}% | gripper {gripper}")
        
        move_save_ee(pose_cmd)


def get_closest_solution(current_q, ik_q):

    # 1. ท่าที่ IK คำนวณมาได้ปกติ
    sol1 = np.array(ik_q)
    
    # 2. ท่าทางที่เป็นอีกทางเลือก (ข้อมือพลิกกลับด้าน)
    # หลักการ: q4 + 180, q5 กลับเครื่องหมาย, q6 + 180
    sol2 = np.copy(sol1)
    sol2[3] = sol2[3] + np.pi if sol2[3] < 0 else sol2[3] - np.pi
    sol2[4] = -sol2[4]
    sol2[5] = sol2[5] + np.pi if sol2[5] < 0 else sol2[5] - np.pi

    # เลือกคำตอบที่ระยะห่างจากท่าทางปัจจุบันน้อยที่สุด
    dist1 = np.linalg.norm(sol1 - current_q)
    dist2 = np.linalg.norm(sol2 - current_q)
    best_sol = np.copy(sol1) if dist1 < dist2 else np.copy(sol2)
    
    # 3. จัดการ Angle Wrap (แก้ปัญหามุมกระโดด -179 <-> 180)
    for i in range(6):
        diff = best_sol[i] - current_q[i]
        # ถ้าสั่งให้หมุนเกินครึ่งรอบ ให้กลับไปหมุนอีกทางที่สั้นกว่า
        if diff > np.pi:
            best_sol[i] -= 2 * np.pi
        elif diff < -np.pi:
            best_sol[i] += 2 * np.pi
            
    return best_sol

# ================= Main Execution =================
if __name__ == '__main__':
    rclpy.init()
    node = JointPublisher()
    
    # รัน ROS Executor ไว้ใน Background Thread
    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()

    jacobian = RobotVelocityKinematics()

    print("✅ Node หุ่นยนต์ทำงานแล้ว (No GUI)")
    print("📡 รอรับคำสั่งตำแหน่งผ่าน Topic: /goto_position")
    
    try:
        # ให้ Main Thread รอไปเรื่อยๆ จนกว่าจะกด Ctrl+C
        while rclpy.ok():
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("🛑 กำลังปิดโปรแกรม...")
    finally:
        node.publish_joints_velo([0.0]*6)
        node.destroy_node()
        rclpy.shutdown()