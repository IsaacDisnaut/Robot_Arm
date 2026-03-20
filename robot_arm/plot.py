import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Button
import json
import numpy as np

# ================= Math & Kinematics =================
def rpy_to_matrix(roll, pitch, yaw):
    Rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx

def get_transform(theta, d, a, alpha):
    return np.array([
        [np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
        [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
        [0,              np.sin(alpha),                np.cos(alpha),               d],
        [0,              0,                            0,                           1]
    ])

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

# ================= ROS 2 Node =================
class PositionUI_Publisher(Node):
    def __init__(self):
        super().__init__('goto_position_ui_node')
        self.publisher_ = self.create_publisher(String, '/goto_position', 10)
        self.get_logger().info('Ready to publish target position with XYZ + RPY -> IK.')

        self.l1 = 0.28787
        self.l2 =  0.26096
        self.l3 = 0.26136
        self.d6 = 0.07074
        self.j4_offset_y = 0

        self.command_data = {
            "sequence": 2, 
            "label": "Move to Cup Dispenser", 
            "j1": 0.0, "j2": 0.0, "j3": 0.0, 
            "j4": 0.0, "j5": 0.0, "j6": 0.0, 
            "rail": 60.0, "speed": 70.0, "gripper": 0, 
            "tcp_x": 0.0, "tcp_y": 0.0, "tcp_z": 0.0,
            "controlMode": "joint" # 🟢 ใส่เผื่อไว้ให้ทำงานเข้ากับโค้ดฝั่งรับ
        }

    def publish_target(self, x, y, z, roll_deg, pitch_deg, yaw_deg):
        local_target_pos = np.array([float(x), float(y), float(z)])
        
        roll_rad = np.radians(float(roll_deg))
        pitch_rad = np.radians(float(pitch_deg))
        yaw_rad = np.radians(float(yaw_deg))
        target_orient = rpy_to_matrix(roll_rad, pitch_rad, yaw_rad)

        q_rad, reachable = inverse_kinematics_6dof(
            local_target_pos, target_orient, self.l1, self.l2, self.l3, self.d6
        )

        if not reachable:
            self.get_logger().warning('⚠️ เป้าหมายนี้อยู่นอกระยะการทำงาน (Unreachable)!')

        q_deg = np.round(np.degrees(q_rad), 2)

        self.command_data['label'] = "Move via IK"
        self.command_data['tcp_x'] = float(x)
        self.command_data['tcp_y'] = float(y)
        self.command_data['tcp_z'] = float(z)
        self.command_data['j1'] = float(q_deg[0])
        self.command_data['j2'] = float(q_deg[1])
        self.command_data['j3'] = float(q_deg[2])
        self.command_data['j4'] = float(q_deg[3])
        self.command_data['j5'] = float(q_deg[4])
        self.command_data['j6'] = float(q_deg[5])
        
        msg = String()
        msg.data = json.dumps(self.command_data) 
        self.publisher_.publish(msg)
        self.get_logger().info(f'Published: XYZ=({x}, {y}, {z}) -> J1-J6 = {q_deg.tolist()}')

    # 🟢 เพิ่มฟังก์ชันสำหรับการส่งค่า Zero
    def publish_zero(self):
        self.command_data['label'] = "Zero All Joints"
        self.command_data['j1'] = 0.0
        self.command_data['j2'] = 0.0
        self.command_data['j3'] = 0.0
        self.command_data['j4'] = 0.0
        self.command_data['j5'] = 0.0
        self.command_data['j6'] = 0.0
        self.command_data['rail'] = 0.0
        self.command_data['speed'] = 40.0 # ตั้งความเร็วปลอดภัยไว้ที่ 40%

        msg = String()
        msg.data = json.dumps(self.command_data) 
        self.publisher_.publish(msg)
        self.get_logger().info('Published: 🔴 ZERO ALL JOINTS (J1-J6=0, Rail=0)')

def main(args=None):
    rclpy.init(args=args)
    node = PositionUI_Publisher()

    fig, ax = plt.subplots(figsize=(6, 7)) # 🟢 ขยายความสูงหน้าต่างขึ้นนิดนึง
    fig.canvas.manager.set_window_title('TCP (XYZ + RPY) & IK (ROS 2)')
    ax.axis('off')

    # 🟢 ปรับลดระยะ Y ลงมาเพื่อเผื่อที่ว่างให้ปุ่มที่ 2
    ax_x = fig.add_axes([0.35, 0.82, 0.45, 0.06])
    ax_y = fig.add_axes([0.35, 0.73, 0.45, 0.06])
    ax_z = fig.add_axes([0.35, 0.64, 0.45, 0.06])
    ax_r = fig.add_axes([0.35, 0.55, 0.45, 0.06])
    ax_p = fig.add_axes([0.35, 0.46, 0.45, 0.06])
    ax_yaw = fig.add_axes([0.35, 0.37, 0.45, 0.06])
    
    ax_btn = fig.add_axes([0.35, 0.22, 0.45, 0.09])
    ax_btn_zero = fig.add_axes([0.35, 0.10, 0.45, 0.09]) # 🟢 เพิ่มกล่องสำหรับปุ่ม Zero

    text_box_x = TextBox(ax_x, 'TCP X (m): ', initial='0.3')
    text_box_y = TextBox(ax_y, 'TCP Y (m): ', initial='0.0')
    text_box_z = TextBox(ax_z, 'TCP Z (m): ', initial='0.3')
    text_box_r = TextBox(ax_r, 'Roll (deg): ', initial='0.0')
    text_box_p = TextBox(ax_p, 'Pitch (deg): ', initial='90.0')
    text_box_yaw = TextBox(ax_yaw, 'Yaw (deg): ', initial='0.0')
    
    btn_publish = Button(ax_btn, 'Calc IK & Publish', color='lightgreen', hovercolor='0.8')
    btn_zero = Button(ax_btn_zero, 'ZERO ALL JOINTS', color='salmon', hovercolor='red') # 🟢 สร้างปุ่ม Zero

    def on_publish_clicked(event):
        try:
            x_val, y_val, z_val = text_box_x.text, text_box_y.text, text_box_z.text
            r_val, p_val, yaw_val = text_box_r.text, text_box_p.text, text_box_yaw.text
            
            node.publish_target(x_val, y_val, z_val, r_val, p_val, yaw_val)
            
            btn_publish.color = 'yellow'
            fig.canvas.draw_idle()
            plt.pause(0.1)
            btn_publish.color = 'lightgreen'
            fig.canvas.draw_idle()

        except ValueError:
            node.get_logger().error("Error: กรุณากรอกตัวเลขเท่านั้น!")
        except Exception as e:
            node.get_logger().error(f"IK Error: {e}")

    # 🟢 Event เมื่อกดปุ่ม Zero
    def on_zero_clicked(event):
        try:
            node.publish_zero()
            
            # Animation เล็กๆ ให้รู้ว่าปุ่มโดนกดแล้ว
            btn_zero.color = 'yellow'
            fig.canvas.draw_idle()
            plt.pause(0.1)
            btn_zero.color = 'salmon'
            fig.canvas.draw_idle()
            
        except Exception as e:
            node.get_logger().error(f"Zero Error: {e}")

    btn_publish.on_clicked(on_publish_clicked)
    btn_zero.on_clicked(on_zero_clicked) # 🟢 ผูก Event เข้ากับปุ่ม
    
    plt.show()

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()