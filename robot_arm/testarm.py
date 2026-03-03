#!/usr/bin/env python3

import sys
import numpy as np
from math import *
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from PyQt5.QtWidgets import *
from PyQt5.QtCore import QTimer
import os
import json
from scipy.optimize import minimize
# -------------------------
# Rotation Utilities
# -------------------------
def rpy_to_matrix(roll, pitch, yaw):
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll),  np.cos(roll)]
    ])

    Ry = np.array([
        [np.cos(pitch),  0, np.sin(pitch)],
        [0,              1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])

    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0,            0,           1]
    ])

    return Rz @ Ry @ Rx


def matrix_to_rpy(R):
    """แปลง Rotation Matrix กลับเป็นมุม Roll, Pitch, Yaw"""
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[0, 0]**2 + R[1, 0]**2))
    
    # เช็ค Singularity (Gimbal lock) เมื่อ Pitch หันขึ้น/ลง 90 องศา
    if np.abs(pitch - np.pi/2) < 1e-6:
        yaw = 0.0
        roll = np.arctan2(R[0, 1], R[0, 2])
    elif np.abs(pitch + np.pi/2) < 1e-6:
        yaw = 0.0
        roll = -np.arctan2(R[0, 1], R[0, 2])
    else:
        yaw = np.arctan2(R[1, 0], R[0, 0])
        roll = np.arctan2(R[2, 1], R[2, 2])
        
    return roll, pitch, yaw


# -------------------------
# 6 DOF Kinematics (DH Parameter Base)
# -------------------------
def dh_matrix(theta, d, a, alpha):
    """ฟังก์ชันสร้าง Transformation Matrix 4x4 จากพารามิเตอร์ DH"""
    return np.array([
        [np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
        [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
        [0,             np.sin(alpha),               np.cos(alpha),              d],
        [0,             0,                           0,                          1]
    ])

def forward_kinematics(q):
    """
    รับค่ามุมข้อต่อ q (list ขนาด 6) หน่วยเรเดียน
    คืนค่า [x, y, z, roll, pitch, yaw] ของ End-effector
    """
    # ตาราง DH ชุดเดียวกับ IK
    dh_table = [
        [q[0] - np.pi/2, 0.28787, 0.02025, -np.pi/2],   
        [q[1] - np.pi/2, 0.0,     0.26097, -np.pi/2],       
        [q[2],           0.0,     0.0179,  -np.pi/2],   
        [q[3],           0.26075, 0.0,      np.pi/2],  
        [q[4],           0.07974, 0.0,     -np.pi/2],   
        [q[5],           0.001,   0.0,      0.0]        
    ]

    T_end_effector = np.eye(4)
    
    # คูณ Transformation Matrix ของแต่ละ Joint
    for row in dh_table:
        T_i = dh_matrix(row[0], row[1], row[2], row[3])
        T_end_effector = T_end_effector @ T_i

    # ดึงค่า X, Y, Z จากคอลัมน์สุดท้าย
    x, y, z = T_end_effector[0, 3], T_end_effector[1, 3], T_end_effector[2, 3]
    
    # ดึง Rotation Matrix แล้วแปลงกลับเป็น RPY
    R = T_end_effector[:3, :3]
    roll, pitch, yaw = matrix_to_rpy(R)
    
    return x, y, z, roll, pitch, yaw

def solve_ik(x, y, z, roll, pitch, yaw):
    """
    Geometric IK 6DOF (3 position + 3 orientation)
    ใช้ DH parameter ชุดเดียวกับ forward_kinematics()
    """

    # ===== DH ค่าคงที่ =====
    d1 = 0.28787
    a2 = 0.26097
    a3 = 0.0179
    d4 = 0.26075
    d5 = 0.07974
    d6 = 0.001

    # ----- 1) หา Wrist Center -----
    R06 = rpy_to_matrix(roll, pitch, yaw)
    p = np.array([-y, x, z])
    wc = p - d6 * R06[:, 2]

    wx, wy, wz = wc

    # ----- 2) Solve q1 -----
    q1 = np.arctan2(wy, wx)

    # ----- 3) Solve q2, q3 -----
    r = np.sqrt(wx**2 + wy**2) - 0.02025
    s = wz - d1

    D = (r**2 + s**2 - a2**2 - d4**2) / (2 * a2 * d4)
    D = np.clip(D, -1.0, 1.0)

    q3 = np.arctan2(np.sqrt(1 - D**2), D)   # elbow-down

    q2 = np.arctan2(s, r) - np.arctan2(d4*np.sin(q3), a2 + d4*np.cos(q3))

    # ----- 4) Orientation part -----
    # คำนวณ R03 จาก DH จริง
    T01 = dh_matrix(q1 - np.pi/2, d1, 0.02025, -np.pi/2)
    T12 = dh_matrix(q2 - np.pi/2, 0, a2, -np.pi/2)
    T23 = dh_matrix(q3, 0, a3, -np.pi/2)

    T03 = T01 @ T12 @ T23
    R03 = T03[:3, :3]

    R36 = R03.T @ R06

    q5 = np.arccos(np.clip(R36[2, 2], -1.0, 1.0))

    if abs(q5) > 1e-6:
        q4 = np.arctan2(R36[1, 2], R36[0, 2])
        q6 = np.arctan2(R36[2, 1], -R36[2, 0])
    else:
        q4 = 0.0
        q6 = np.arctan2(-R36[0, 1], R36[0, 0])

    return [q1- np.pi/2, q2- np.pi/2, q3- np.pi/2, -q4, q5, -q6]

# -------------------------
# ROS2 Publisher
# -------------------------
class IKPublisher(Node):
    def __init__(self):
        super().__init__('ik_6dof_gui')
        self.pub = self.create_publisher(JointState, '/joint_states', 10)

        self.names = [
            "joint_1","joint_2","joint_3",
            "joint_4","joint_5","joint_6","slider_joint"
        ]
        
        self.current_q = [0.0] * 7
        self.target_q = [0.0] * 7
        self.step_size = 0.02  

    def set_target(self, q):
        self.target_q = list(q)

    def interpolate_and_publish(self):
        easing_factor = 0.025  
        tolerance = 0.001    

        for i in range(7):
            diff = self.target_q[i] - self.current_q[i]
            if abs(diff) > tolerance:
                self.current_q[i] += diff * easing_factor
            else:
                self.current_q[i] = self.target_q[i]

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.names
        msg.position = self.current_q
        self.pub.publish(msg)


# -------------------------
# หน้าต่าง Pop-up สำหรับโชว์และจัดการ Task
# -------------------------
class TaskListPopup(QDialog):
    def __init__(self, parent=None, node=None):
        super().__init__(parent)
        self.node = node
        self.setWindowTitle("Saved Tasks Management")
        self.resize(350, 400) 
        
        layout = QVBoxLayout()
        
        self.list_widget = QListWidget()
        layout.addWidget(self.list_widget)
        
        btn_layout = QHBoxLayout()
        
        self.btn_play = QPushButton("▶ Play Task")
        self.btn_play.clicked.connect(self.play_selected_task)
        btn_layout.addWidget(self.btn_play)
        
        self.btn_delete = QPushButton("🗑 Delete Task")
        self.btn_delete.clicked.connect(self.delete_selected_task)
        btn_layout.addWidget(self.btn_delete)
        
        layout.addLayout(btn_layout)
        
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        layout.addWidget(btn_close)
        
        self.setLayout(layout)
        self.load_tasks()

    def load_tasks(self):
        self.list_widget.clear() 
        path = '/home/isaac/ros2_ws/src/robot_arm/tasks/data.txt'
        
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if not data:
                        self.list_widget.addItem("ยังไม่มี Task (ไฟล์ว่างเปล่า)")
                        self.btn_play.setEnabled(False)
                        self.btn_delete.setEnabled(False)
                        return
                    
                    self.btn_play.setEnabled(True)
                    self.btn_delete.setEnabled(True)

                    for task in data:
                        task_no = task.get("task_no", "?")
                        num_jogs = len([k for k in task.keys() if k.startswith("jog")])
                        self.list_widget.addItem(f"Task {task_no} (มี {num_jogs} jogs)")
            except json.JSONDecodeError:
                self.list_widget.addItem("ไม่สามารถอ่านข้อมูล Task ได้")
        else:
            self.list_widget.addItem("ยังไม่มีข้อมูล Task บันทึกไว้")
            self.btn_play.setEnabled(False)
            self.btn_delete.setEnabled(False)

    def play_selected_task(self):
        selected_item = self.list_widget.currentItem()
        if not selected_item:
            QMessageBox.warning(self, "แจ้งเตือน", "กรุณาเลือก Task ที่ต้องการรันก่อนครับ")
            return

        task_text = selected_item.text()
        try:
            task_no = int(task_text.split(" ")[1])
        except ValueError:
            return

        path = '/home/isaac/ros2_ws/src/robot_arm/tasks/data.txt'
        target_task = None
        
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for t in data:
                    if t.get("task_no") == task_no:
                        target_task = t
                        break

        if target_task:
            jogs = []
            jog_count = len([k for k in target_task.keys() if k.startswith("jog")])
            
            for i in range(1, jog_count + 1):
                jog_key = f"jog{i}"
                if jog_key in target_task:
                    jogs.append(target_task[jog_key])
            
            main_window = self.parent()
            main_window.start_playback(jogs)
            self.accept()

    def delete_selected_task(self):
        selected_item = self.list_widget.currentItem()
        if selected_item:
            print(f"กำลังจะลบ: {selected_item.text()}")
            # TODO: ใส่โค้ดลบ Task ออกจากไฟล์ JSON ตรงนี้
        else:
            QMessageBox.warning(self, "แจ้งเตือน", "กรุณาเลือก Task ที่ต้องการลบก่อนครับ")


# -------------------------
# GUI หน้าต่างหลัก
# -------------------------
class IKWindow(QWidget):
    def __init__(self,node):
        super().__init__()
        self.node = node
        self.setWindowTitle("6 DOF IK Controller")
        
        layout = QVBoxLayout()
        self.iteration = 0
        self.inputs = {}
        labels = ["X","Y","Z","Roll","Pitch","Yaw","Slider"]
       
        for l in labels:
            layout.addWidget(QLabel(l))
            inp = QLineEdit()
            inp.setText("0")                       
            inp.editingFinished.connect(self.ensure_not_empty)
            inp.editingFinished.connect(self.compute)
            layout.addWidget(inp)
            self.inputs[l] = inp

        btn = QPushButton("Compute IK")
        btn.clicked.connect(self.compute)  
        layout.addWidget(btn)

        self.current_jogs = []       
        self.last_saved_pose = None  

        btn_save_pos = QPushButton("Save Position")
        btn_save_pos.clicked.connect(self.add_iter)
        layout.addWidget(btn_save_pos)

        btn_save_task = QPushButton("Save Task")
        btn_save_task.clicked.connect(self.tasksave)
        layout.addWidget(btn_save_task)

        btn_show_tasks = QPushButton("Show Saved Tasks")
        btn_show_tasks.clicked.connect(self.show_task_popup)
        layout.addWidget(btn_show_tasks)

        self.status = QLabel("")
        layout.addWidget(self.status)
        
        self.setLayout(layout)
        self.compute()

    def show_task_popup(self):
        popup = TaskListPopup(self, self.node)
        popup.exec_()

    def add_iter(self):
        try:
            x = float(self.inputs["X"].text())
            y = float(self.inputs["Y"].text())
            z = float(self.inputs["Z"].text())
            roll = float(self.inputs["Roll"].text())
            pitch = float(self.inputs["Pitch"].text())
            yaw = float(self.inputs["Yaw"].text())
            slider = float(self.inputs["Slider"].text())

            current_pose = [x, y, z, roll, pitch, yaw,slider]

            if current_pose != self.last_saved_pose:
                self.current_jogs.append(current_pose)
                self.last_saved_pose = current_pose
                jog_number = len(self.current_jogs)
                
                msg = f"✅ บันทึก Position แล้ว (jog{jog_number}): {current_pose}"
                self.status.setText(msg)
                print(msg)
            else:
                msg = "⚠️ ตำแหน่งไม่ได้เปลี่ยนแปลง ข้ามการบันทึกซ้ำ"
                self.status.setText(msg)
                print(msg)

        except ValueError:
            self.status.setText("❌ เกิดข้อผิดพลาด: กรุณาตรวจสอบช่อง Input ว่าเป็นตัวเลข")

    def ensure_not_empty(self):
        for inp in self.inputs.values():
            if inp.text().strip() == "":
                inp.setText("0")

    def compute(self):
        try:
            # รับค่าพิกัดเป้าหมายจาก GUI
            x = float(self.inputs["X"].text())
            y = float(self.inputs["Y"].text())
            z = float(self.inputs["Z"].text())

            roll = np.radians(float(self.inputs["Roll"].text()))
            pitch = np.radians(float(self.inputs["Pitch"].text()))
            yaw = np.radians(float(self.inputs["Yaw"].text()))
            slider = float(self.inputs["Slider"].text())
            
            # 1. คำนวณ Inverse Kinematics เพื่อหามุมข้อต่อ
            q = solve_ik(x, y, z, roll, pitch, yaw)
            
            # เก็บค่าส่งไป ROS2
            target_joints = q + [slider]
            self.node.set_target(target_joints)

            # 2. นำมุมที่ได้ กลับไปเข้าสมการ Forward Kinematics เพื่อยืนยันพิกัดปลายแขน
            fk_x, fk_y, fk_z, fk_roll, fk_pitch, fk_yaw = forward_kinematics(q)

            # แปลงมุมสำหรับแสดงผล
            q_deg = [round(np.degrees(a), 2) for a in q]
            fk_r_deg = np.degrees(fk_roll)
            fk_p_deg = np.degrees(fk_pitch)
            fk_y_deg = np.degrees(fk_yaw)

            # สร้างข้อความสำหรับ Status Label (แสดงทั้ง Joint และ FK ผลลัพธ์)
            status_text = (
                f"🎯 Joints (deg): {q_deg} | Slider: {round(slider, 3)} m\n"
                f"✅ FK Verify   : X={fk_x:.3f}, Y={fk_y:.3f}, Z={fk_z:.3f} | "
                f"R={fk_r_deg:.1f}°, P={fk_p_deg:.1f}°, Y={fk_y_deg:.1f}°"
            )
            self.status.setText(status_text)

        except Exception as e:
            self.status.setText(f"Error: {str(e)}")

    def tasksave(self):
        if not self.current_jogs:
            self.status.setText("⚠️ ไม่มี Position ให้บันทึก กรุณากด Save Position อย่างน้อย 1 ครั้ง")
            return

        path = '/home/isaac/ros2_ws/src/robot_arm/tasks/data.txt'
        existing_data = []
        task_no = 1

        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                try:
                    existing_data = json.load(f)
                    if len(existing_data) > 0:
                        last_task = existing_data[-1].get("task_no", 0)
                        task_no = last_task + 1
                except json.JSONDecodeError:
                    pass 

        if len(existing_data) >= 10:
            msg = "❌ ไม่สามารถเพิ่ม Task ใหม่ได้: ข้อมูลเต็มความจุแล้ว (สูงสุด 10 Task)"
            self.status.setText(msg)
            print(msg)
            return

        new_task = {"task_no": task_no}
        
        for i, pose in enumerate(self.current_jogs):
            key_name = f"jog{i + 1}"
            new_task[key_name] = pose
            
        existing_data.append(new_task)

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as file:
            json.dump(existing_data, file, ensure_ascii=False, indent=4)

        success_msg = f"✅ บันทึก Task ที่ {task_no} เรียบร้อย! (จำนวน {len(self.current_jogs)} jogs)"
        self.status.setText(success_msg)
        print(success_msg)

        self.current_jogs = []
        self.last_saved_pose = None
        
        self.show_task_popup()

    def start_playback(self, jogs):
        self.playback_jogs = jogs
        self.playback_index = 0
        
        self.playback_timer = QTimer(self)
        self.playback_timer.timeout.connect(self.play_next_jog)
        self.playback_timer.start(2500) 
        
        self.play_next_jog()

    def play_next_jog(self):
        if self.playback_index < len(self.playback_jogs):
            pose = self.playback_jogs[self.playback_index]
            
            self.inputs["X"].setText(str(pose[0]))
            self.inputs["Y"].setText(str(pose[1]))
            self.inputs["Z"].setText(str(pose[2]))
            self.inputs["Roll"].setText(str(pose[3]))
            self.inputs["Pitch"].setText(str(pose[4]))
            self.inputs["Yaw"].setText(str(pose[5]))
            self.inputs["Slider"].setText(str(pose[6]))
            
            self.compute()
            
            self.playback_index += 1
            self.status.setText(f"▶ กำลังเล่น Task... (จุดที่ {self.playback_index}/{len(self.playback_jogs)})")
        else:
            self.playback_timer.stop()
            self.status.setText("✅ เล่น Task จบสมบูรณ์!")

def main():
    rclpy.init()
    node = IKPublisher()

    app = QApplication(sys.argv)
    window = IKWindow(node)
    window.show()

    timer = QTimer()
    
    def update_loop():
        rclpy.spin_once(node, timeout_sec=0)  
        node.interpolate_and_publish()        
        
    timer.timeout.connect(update_loop)
    timer.start(20)  

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()