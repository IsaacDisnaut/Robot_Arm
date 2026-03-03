import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons, Button # 🔥 เพิ่ม Button เข้ามา
from mpl_toolkits.mplot3d import Axes3D
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import threading
import os
import json
import time

TASK_FILE_PATH = '/home/isaac/ros2_ws/src/robot_arm/tasks/data.txt'
L1, L2, L3 = 0.28787, 0.26096, 0.26136
D6 = 0.07074
# 🔥 กำหนดระยะ Offset ของ J4 สำหรับวาดกราฟ
J4_OFFSET_Y = 0.02175 
class RobotVelocityKinematics:
    def __init__(self):
        # คุณสามารถเก็บค่าพารามิเตอร์คงที่ของหุ่นยนต์ไว้ตรงนี้ได้ในอนาคต
        pass

    def _dh_matrix(self, theta, d, a, alpha):
        """(Private) สร้าง Transformation Matrix 4x4 จากพารามิเตอร์ DH"""
        return np.array([
            [np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
            [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
            [0,             np.sin(alpha),               np.cos(alpha),              d],
            [0,             0,                           0,                          1]
        ])

    def get_jacobian(self,q):
        """
        คำนวณ Geometric Jacobian Matrix (ขนาด 6x6)
        แถว 1-3: ความเร็วเชิงเส้น (Vx, Vy, Vz)
        แถว 4-6: ความเร็วเชิงมุม (Wx, Wy, Wz)
        """
        l1, l2, l3 = 0.28787, 0.26096, 0.26136
        d6 = 0.07074
        # 🔥 กำหนดระยะ Offset ของ J4 สำหรับวาดกราฟ
        offset_y = 0.02175 
        a1 = 0.020885
    # คำนวณมุมและระยะทแยงที่เกิดจากการยก J4 สูงขึ้น
        gamma = np.arctan2(offset_y, l3)
        l3_eff = np.sqrt(l3**2 + offset_y**2)
        q1,q2,q3,q4,q5,q6 = q
        # ตาราง DH ชุดเดียวกับหุ่นยนต์ของคุณ
        dh_table = [
        [q1,           l1, a1,  -np.pi/2], 
        [q2,           0,  l2,  0      ], 
        # 🔥 กราฟิก: J3 หมุนเงยขึ้นหลบโครงสร้าง (ใส่เครื่องหมายลบ)
        [q3 + np.pi/2 - gamma, 0,  0,   np.pi/2], 
        # 🔥 กราฟิก: ระยะ L3 ยืดออกเป็นเส้นทแยงมุม
        [q4,           l3_eff, 0,  -np.pi/2], 
        [q5+gamma,     0,  0,   np.pi/2], 
        [q6,           d6, 0,   0      ]  
    ]

        # สะสมเมทริกซ์การแปลงพิกัด
        T_matrices = [np.eye(4)]
        T = np.eye(4)
        for row in dh_table:
            T = T @ self._dh_matrix(*row)
            T_matrices.append(T)

        # พิกัดของปลายมือ (End-effector)
        p_e = T_matrices[-1][0:3, 3]

        J = np.zeros((6, 6))

        for i in range(6):
            z_i = T_matrices[i][0:3, 2]  # แกนหมุน
            p_i = T_matrices[i][0:3, 3]  # จุดกำเนิดข้อต่อ

            # ครึ่งบน: Linear Velocity (Vx, Vy, Vz)
            J[0:3, i] = np.cross(z_i, (p_e - p_i))
            # ครึ่งล่าง: Angular Velocity (Wx, Wy, Wz)
            J[3:6, i] = z_i

        return J

    def forward_velocity(self, q, q_dot):
        """
        รับค่า: มุมปัจจุบัน (q) และ ความเร็วมอเตอร์ (q_dot) [ขนาด 6x1]
        คืนค่า: ความเร็วปลายมือ [Vx, Vy, Vz, Wx, Wy, Wz]
        """
        J = self.get_jacobian(q)
        return J @ np.array(q_dot)

    def inverse_velocity(self, q, target_velocity):
        """
        รับค่า: มุมปัจจุบัน (q) และ ความเร็วปลายมือที่ต้องการ [Vx, Vy, Vz, Wx, Wy, Wz]
        คืนค่า: ความเร็วมอเตอร์ที่ต้องสั่ง (q_dot) [ขนาด 6x1]
        """
        J = self.get_jacobian(q)
        
        # ใช้ Pseudo-inverse ป้องกัน Singularity
        J_pinv = np.linalg.pinv(J)
        return J_pinv @ np.array(target_velocity)

# ================= ROS2 Node =================
class JointPublisher(Node):
    def __init__(self):
        super().__init__('ik_joint_publisher')
        self.publisher = self.create_publisher(JointState, 'joint_states', 10)
        self.subscription = self.create_subscription(
            JointState,
            'joint_states',
            self.joint_cb,
            10)
        
        # 1. เพิ่มตัวแปรสำหรับเก็บค่า Feedback จาก Subscriber
        self.current_joint_positions = [0.0] * 6
        self.current_slider_position = 0.0
        self.has_received_data = False

        # 2. เพิ่มตัวแปรสำหรับ "พักข้อมูล" ก่อนให้ Timer ส่ง
        self.joints_to_publish = [0.0] * 6
        self.rail_to_publish = 0.0
        self.velocity_to_publish = [0.0] * 7 # (slider 1 ตัว + joints 6 ตัว)

        # 3. สร้าง Timer ให้พ่นข้อมูลทุก 0.02 วินาที (50Hz)
        self.timer = self.create_timer(0.02, self.timer_callback)
        
    def publish_joints(self, joints, base_y):
        """รับคำสั่งแบบ Position (ไม่ต้อง publish ตรงๆ แล้ว ให้แค่บันทึกค่า)"""
        self.joints_to_publish = [float(q) for q in joints]
        self.rail_to_publish = float(base_y)
        # ถ้าสั่ง position แปลว่าไม่ได้สั่ง velocity ให้เคลียร์เป็น 0
        self.velocity_to_publish = [0.0] * 7

    def publish_joints_velo(self, joints):
        """รับคำสั่งแบบ Velocity (ไม่ต้อง publish ตรงๆ แล้ว ให้แค่บันทึกค่า)"""
        # joints ที่รับมามี 6 ตัว (q1-q6) เราเติม 0.0 ไว้ข้างหน้าสำหรับ slider
        self.velocity_to_publish = [0.0] + [float(q) for q in joints]

    def joint_cb(self, msg):
        """รับค่าจากหุ่นยนต์จริง (Feedback)"""
        if len(msg.position) >= 7:
            self.current_slider_position = msg.position[0]
            self.current_joint_positions = list(msg.position[1:7])
            self.has_received_data = True
            
            # (ทางเลือก) ถ้าเพิ่งเปิดโปรแกรม ให้ล็อกค่าเริ่มต้นไว้ที่หุ่นจริงป้องกันหุ่นกระชากไปที่ 0
            if not any(self.joints_to_publish): 
                self.joints_to_publish = list(msg.position[1:7])
                self.rail_to_publish = msg.position[0]

    def timer_callback(self):
        """ฟังก์ชันเดียวที่ทำหน้าที่ Publish ออกสู่โลกภายนอก"""
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = ['slider_joint', 'joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']
        
        # 🟢 ลอจิก Integration 
        # เช็คก่อนว่ามีการสั่งความเร็วเข้ามาหรือไม่ (ถ้าตัวแปร velocity ไม่ใช่ 0)
        # ถ้ามีความเร็ว ให้ค่อยๆ บวกเข้าไปใน Position 
        dt = 0.02
        if any(v != 0.0 for v in self.velocity_to_publish):
            self.rail_to_publish += self.velocity_to_publish[0] * dt
            for i in range(6):
                self.joints_to_publish[i] += self.velocity_to_publish[i+1] * dt 

        # จัดเตรียมข้อมูลส่ง
        msg.position = [float(self.rail_to_publish)] + [float(q) for q in self.joints_to_publish]
        msg.velocity = [float(v) for v in self.velocity_to_publish]
        
        self.publisher.publish(msg)

# ================= Robot Params =================

jacobian = RobotVelocityKinematics()
# ================= Math & Kinematics =================
def rpy_to_matrix(roll, pitch, yaw):
    Rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx

def matrix_to_rpy(T):
    """แปลง Transformation Matrix เป็น x, y, z, roll, pitch, yaw"""
    # ดึงพิกัด x, y, z
    x = T[0, 3]
    y = T[1, 3]
    z = T[2, 3]
    
    # ดึงเมทริกซ์การหมุน (Rotation Matrix 3x3)
    R = T[:3, :3]
    
    # คำนวณ Pitch (รอบแกน Y)
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[0, 0]**2 + R[1, 0]**2))
    
    # เช็คสภาวะ Gimbal Lock (Pitch +- 90 องศา)
    if np.abs(pitch - np.pi/2) < 1e-6:
        yaw = 0.0
        roll = np.arctan2(R[0, 1], R[0, 2])
    elif np.abs(pitch + np.pi/2) < 1e-6:
        yaw = 0.0
        roll = -np.arctan2(R[0, 1], R[0, 2])
    else:
        # สภาวะปกติ
        yaw = np.arctan2(R[1, 0], R[0, 0])
        roll = np.arctan2(R[2, 1], R[2, 2])
        
    # แปลงจากเรเดียนเป็นองศา
    roll_deg = np.degrees(roll)
    pitch_deg = np.degrees(pitch)
    yaw_deg = np.degrees(yaw)
    
    return x, y, z, roll_deg, pitch_deg, yaw_deg

def get_transform(theta, d, a, alpha):
    return np.array([
        [np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
        [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
        [0,              np.sin(alpha),                np.cos(alpha),               d],
        [0,              0,                            0,                           1]
    ])

# 🔥 ฟังก์ชันที่ใช้วาดกราฟิกโดยเฉพาะ (มี J4 Offset + Base Track Y)
def forward_kinematics_visual(q, l1, l2, l3, d6, offset_y, base_y=0.0):
    q1, q2, q3, q4, q5, q6 = q
    a1 = 0.020885
    # คำนวณมุมและระยะทแยงที่เกิดจากการยก J4 สูงขึ้น
    gamma = np.arctan2(offset_y, l3)
    l3_eff = np.sqrt(l3**2 + offset_y**2)
    
    dh_params = [
        [q1,           l1, a1,  -np.pi/2], 
        [q2,           0,  l2,  0      ], 
        # 🔥 กราฟิก: J3 หมุนเงยขึ้นหลบโครงสร้าง (ใส่เครื่องหมายลบ)
        [q3 + np.pi/2 - gamma, 0,  0,   np.pi/2], 
        # 🔥 กราฟิก: ระยะ L3 ยืดออกเป็นเส้นทแยงมุม
        [q4,           l3_eff, 0,  -np.pi/2], 
        [q5+gamma,     0,  0,   np.pi/2], 
        [q6,           d6, 0,   0      ]  
    ]
    
    T = np.eye(4)
    # 🔥 เลื่อนฐานหุ่นยนต์ไปตามรางแกน Y
    T[1, 3] = base_y 
    
    T_list = [T.copy()]
    for params in dh_params:
        T = T @ get_transform(*params)
        T_list.append(T.copy())
    return T_list

# ฟังก์ชัน IK หลัก (ไม่มีออฟเซ็ต คลีน 100%)
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


# ================= ROS Spin Thread =================
rclpy.init()
node = JointPublisher()
ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
ros_thread.start()

# ================= Plot & UI Setup =================
fig = plt.figure(figsize=(12, 9))
plt.subplots_adjust(left=0.2, bottom=0.40)
ax = fig.add_subplot(111, projection='3d')

init_x, init_y, init_z = (L2 + L3 + D6), 0.0, L1
init_roll, init_pitch, init_yaw = 0, 90, 0
init_base_y = 0.0

ax_base_y = plt.axes([0.25, 0.33, 0.65, 0.03])
ax_x      = plt.axes([0.25, 0.29, 0.65, 0.03])
ax_y      = plt.axes([0.25, 0.25, 0.65, 0.03])
ax_z      = plt.axes([0.25, 0.21, 0.65, 0.03])
ax_roll   = plt.axes([0.25, 0.17, 0.65, 0.03])
ax_pitch  = plt.axes([0.25, 0.13, 0.65, 0.03])
ax_yaw    = plt.axes([0.25, 0.09, 0.65, 0.03])

s_base_y = Slider(ax_base_y, 'Base Track Y', -1.0, 1.0, valinit=init_base_y, color='orange')
s_x = Slider(ax_x, 'Local X', -1.0, 1.0, valinit=init_x)
s_y = Slider(ax_y, 'Local Y', -1.0, 1.0, valinit=init_y)
s_z = Slider(ax_z, 'Local Z', 0.0, 1.5, valinit=init_z)
s_roll = Slider(ax_roll, 'Roll (deg)', -180, 180, valinit=init_roll)
s_pitch = Slider(ax_pitch, 'Pitch (deg)', -180, 180, valinit=init_pitch)
s_yaw = Slider(ax_yaw, 'Yaw (deg)', -180, 180, valinit=init_yaw)

sliders = [s_base_y, s_x, s_y, s_z, s_roll, s_pitch, s_yaw]

# ================= 🎛️ ปุ่ม (Button) =================
# สร้างพื้นที่สำหรับปุ่ม (x, y, กว้าง, สูง)
ax_button_reset = plt.axes([0.02, 0.35, 0.12, 0.05])
ax_button_save = plt.axes([0.02, 0.20, 0.12, 0.05])
# สร้างปุ่ม
btn_reset = Button(ax_button_reset, 'RESET HOME', color='salmon', hovercolor='red')
btn_run = Button(ax_button_save, 'Run Position', color='green', hovercolor='lime')

def reset_home(event):
    # 1. จัดการเป้าหมาย (Targets)
    
    tar_q = [0.0,-1.5708,1.5708,0,0,0]
    target_slider = 0    
    speed_pct = 40       
    
    # 2. ตั้งค่า PD Gain (ต้องปรับจูนตามความเหมาะสมของหุ่นจริง)
    # Kp: ยิ่งเยอะยิ่งวิ่งเร็วเข้าหาเป้าหมาย
    # Kd: ช่วยเบรก ลดการสะบัด/สั่นเมื่อใกล้ถึงจุดหยุด
    Kp = 5.0 * (speed_pct / 100.0)
    Kd = 0.1
    
    dt = 0.05 # แนะนำ 0.01 (100Hz) หรือ 0.05 (20Hz) สำหรับ Matplotlib
    prev_error_q = np.zeros(6)
    
    print(f"🚀 เคลื่อนที่ด้วย PD Control (Kp: {Kp}, Kd: {Kd})")

    # 4. ลูปควบคุม
    while True:
        current_q = np.array(node.current_joint_positions) 
        # current_slider = node.current_slider_position

        # ข. คำนวณ Error ของ Joint Position
        error_q = tar_q - current_q
        
        # ค. คำนวณความแตกต่างของ Error (Derivative)
        derivative_q = (error_q - prev_error_q) / dt
        
        # ง. กฎการควบคุม PD (PD Control Law)
        q_dot = (Kp * error_q) + (Kd * derivative_q)
        
        # จ. จำกัดความเร็วสูงสุด (Safety Limit)
        max_limit = 1.5 # rad/s
        q_dot = np.clip(q_dot, -max_limit, max_limit)

        # 🟢 1. เช็คว่าแกนไหนถึงเป้าหมายแล้วบ้าง (ค่า error เป็นบวกหรือลบก็ต้องน้อยกว่า 0.005)
        reached_mask = np.abs(error_q) < 0.001
        
        # 🟢 2. บังคับให้แกนที่ถึงแล้ว มีความเร็วเป็น 0 ทันที (แกนอื่นที่ยังไม่ถึงก็วิ่งต่อไป)
        q_dot[reached_mask] = 0.0

        # ช. ส่งค่าความเร็วไปที่ ROS 2
        node.publish_joints_velo(q_dot.tolist())
        
        # เก็บค่า Error ไว้ใช้ในรอบถัดไป
        prev_error_q = error_q
        print(f"Error: {error_q}")
        print(f"Velo : {q_dot}")
        
        if np.all(reached_mask):
            T_cur_list = forward_kinematics_visual(node.current_joint_positions, L1, L2, L3, D6, J4_OFFSET_Y, node.current_slider_position)
            x_c, y_c, z_c, r_c, p_c, yw_c = matrix_to_rpy(T_cur_list[-1])
            # ข. อัปเดตตัวเลข Slider ของพิกัด End-Effector
            s_x.set_val(x_c)
            s_y.set_val(y_c)
            s_z.set_val(z_c)
            s_roll.set_val(r_c)
            s_pitch.set_val(p_c)
            s_yaw.set_val(yw_c)
            print("🎯 ทุกแกนเข้าสู่ตำแหน่งเป้าหมายแล้ว!")
            break
        
            
        plt.pause(dt)
        
    # เมื่อออกจากลูปแล้ว ส่งคำสั่งหยุดสนิทอีกครั้งเพื่อความชัวร์
    node.publish_joints_velo([0.0]*6)

    # 5. จบการทำงาน
    node.publish_joints_velo([0.0]*6) # สั่งหยุดสนิท
    print("✅ ถึงตำแหน่งเป้าหมายด้วย PD Control เรียบร้อยแล้ว")

def move_save(task):
    # 1. จัดการเป้าหมาย (Targets)
    tar_q = np.radians(task[:6])  
    target_slider = task[6]     
    speed_pct = task[7]         
    
    # 2. ตั้งค่า PD Gain (ต้องปรับจูนตามความเหมาะสมของหุ่นจริง)
    # Kp: ยิ่งเยอะยิ่งวิ่งเร็วเข้าหาเป้าหมาย
    # Kd: ช่วยเบรก ลดการสะบัด/สั่นเมื่อใกล้ถึงจุดหยุด
    Kp = 5.0 * (speed_pct / 100.0)
    Kd = 0.1
    
    dt = 0.05 # แนะนำ 0.01 (100Hz) หรือ 0.05 (20Hz) สำหรับ Matplotlib
    prev_error_q = np.zeros(6)
    
    print(f"🚀 เคลื่อนที่ด้วย PD Control (Kp: {Kp}, Kd: {Kd})")

    # 4. ลูปควบคุม
    while True:
        current_q = np.array(node.current_joint_positions) 
        # current_slider = node.current_slider_position

        # ข. คำนวณ Error ของ Joint Position
        error_q = tar_q - current_q
        
        # ค. คำนวณความแตกต่างของ Error (Derivative)
        derivative_q = (error_q - prev_error_q) / dt
        
        # ง. กฎการควบคุม PD (PD Control Law)
        q_dot = (Kp * error_q) + (Kd * derivative_q)
        
        # จ. จำกัดความเร็วสูงสุด (Safety Limit)
        max_limit = 1.5 # rad/s
        q_dot = np.clip(q_dot, -max_limit, max_limit)

        # 🟢 1. เช็คว่าแกนไหนถึงเป้าหมายแล้วบ้าง (ค่า error เป็นบวกหรือลบก็ต้องน้อยกว่า 0.005)
        reached_mask = np.abs(error_q) < 0.005
        
        # 🟢 2. บังคับให้แกนที่ถึงแล้ว มีความเร็วเป็น 0 ทันที (แกนอื่นที่ยังไม่ถึงก็วิ่งต่อไป)
        q_dot[reached_mask] = 0.0

        # ช. ส่งค่าความเร็วไปที่ ROS 2
        node.publish_joints_velo(q_dot.tolist())
        
        # เก็บค่า Error ไว้ใช้ในรอบถัดไป
        prev_error_q = error_q
        print(f"Error: {error_q}")
        print(f"Velo : {q_dot}")
        
        if np.all(reached_mask):
            print("🎯 ทุกแกนเข้าสู่ตำแหน่งเป้าหมายแล้ว!")
            break
        
            
        plt.pause(dt)
        
    # เมื่อออกจากลูปแล้ว ส่งคำสั่งหยุดสนิทอีกครั้งเพื่อความชัวร์
    node.publish_joints_velo([0.0]*6)

    # 5. จบการทำงาน
    node.publish_joints_velo([0.0]*6) # สั่งหยุดสนิท
    print("✅ ถึงตำแหน่งเป้าหมายด้วย PD Control เรียบร้อยแล้ว")

def run_pose(event):
    path = '/home/isaac/ros2_ws/src/robot_arm/tasks/data.txt'
    target_task = None
        
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)
                
                # 🔥 เช็คความปลอดภัย: ต้องเป็น dict และมี key ชื่อ "tasks" เท่านั้นถึงจะใช้ต่อ
                if isinstance(loaded_data, dict) and "tasks" in loaded_data:
                    data = loaded_data
                    
                else:
                    print("⚠️ Unknown format")
        except json.JSONDecodeError:
            pass
        task_list = data.get('tasks',[])
        if not task_list:
            print("⚠️ โปรแกรมนี้ยังไม่มีข้อมูล Task")
            return
        else:
            seq=len(task_list)
            print(f"There're {seq} sequence")
            
            for i in task_list:
                task=[]
                print(f"seq: {i.get("sequence")} name:{i.get("label")}")
                for j in range(1,7):
                    task.append(i.get(f"j{j}"))
                    print(f"moving q{j} to {task[j-1]}")
                task.append(i.get("rail"))
                task.append(i.get("speed"))
                print(f"moving rail to {task[6]} with speed {task[7]} %")
                move_save(task)
                time.sleep(0.001)
                reset_home(event)

    
        
# ผูกฟังก์ชันเข้ากับการคลิกปุ่ม
btn_reset.on_clicked(reset_home)
btn_run.on_clicked(run_pose)
# ==================================================

labels = ["Base", "J1", "J2", "J3", "J4", "J5", "J6"]
axes_visibility = [True] * 7 

ax_check = plt.axes([0.02, 0.45, 0.12, 0.3], facecolor='#f0f0f0')
check = CheckButtons(ax_check, labels, axes_visibility)

def toggle_axes(label):
    idx = labels.index(label)
    axes_visibility[idx] = not axes_visibility[idx]
    update(None)

check.on_clicked(toggle_axes)

def draw_axes(ax, T, length=0.08, label=""):
    origin = T[:3, 3]
    x_axis = T[:3, 0] * length
    y_axis = T[:3, 1] * length
    z_axis = T[:3, 2] * length

    ax.quiver(*origin, *x_axis, color='r', linewidth=1.5)
    ax.quiver(*origin, *y_axis, color='g', linewidth=1.5)
    ax.quiver(*origin, *z_axis, color='b', linewidth=3.0) 
    
    if label:
        offset_z = 0.02
        if label in ["J3", "J5"]: offset_z = -0.04 
        ax.text(origin[0], origin[1], origin[2] + offset_z, label, fontsize=9, fontweight='bold', color='black')

def update(val):
    ax.cla()

    base_y = s_base_y.val
    
    local_target_pos = np.array([s_x.val, s_y.val, s_z.val])
    r, p, y = np.radians(s_roll.val), np.radians(s_pitch.val), np.radians(s_yaw.val)
    target_orient = rpy_to_matrix(r, p, y)

    # คำนวณ IK ด้วยสมการบริสุทธิ์
    joints, reachable = inverse_kinematics_6dof(local_target_pos, target_orient, L1, L2, L3, D6)
    # 🔥 คำนวณกราฟิก FK ที่รวม J4 Offset + Track Y เลื่อนฐาน
    T_list = forward_kinematics_visual(joints, L1, L2, L3, D6, J4_OFFSET_Y, base_y)
    pts = np.array([T[:3, 3] for T in T_list])

    ax.plot([0, 0], [-1.0, 1.0], [0, 0], '--', color='black', linewidth=3, alpha=0.5, label="Linear Track")
    ax.plot(pts[:,0], pts[:,1], pts[:,2], '-o', color='#34495e', linewidth=4, alpha=0.8)

    global_target_pos = local_target_pos + np.array([0, base_y, 0])
    target_color = 'green' if reachable else 'red'
    ax.scatter(global_target_pos[0], global_target_pos[1], global_target_pos[2], color=target_color, s=100, label="Target (Local)")
    ax.scatter(pts[-1,0], pts[-1,1], pts[-1,2], color='purple', s=50, label="Visual Output (Offset)")

    for i, T in enumerate(T_list):
        if axes_visibility[i]:
            draw_axes(ax, T, length=0.08, label=labels[i])

    info_text = "Target is UNREACHABLE!" if not reachable else ""
    ax.text2D(0.05, 0.95, info_text, transform=ax.transAxes, color='red', fontsize=14, fontweight='bold')

    q_deg = np.degrees(joints)
    angle_text = f"Track Y: {base_y:.2f} m\n\nClean Joints (to ROS):\n"
    for i in range(6):
        angle_text += f"q{i+1}: {joints[i]:.2f} rad ({q_deg[i]:.1f}°)\n"
    ax.text2D(0.02, 0.65, angle_text, transform=ax.transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

    node.publish_joints(joints, base_y)

    ax.set_xlim([-1.0, 1.0]); ax.set_ylim([-1.0, 1.0]); ax.set_zlim([0, 1.5])
    ax.set_xlabel('Global X'); ax.set_ylabel('Global Y'); ax.set_zlabel('Global Z')
    ax.legend(loc="upper right")
    ax.set_title("7-Axis System: Base Track + J4 Visual Offset")

    fig.canvas.draw_idle()

for s in sliders:
    s.on_changed(update)


    
update(None)
plt.show()

rclpy.shutdown()