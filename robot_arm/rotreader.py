import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import CheckButtons # เพิ่ม Import สำหรับสร้าง Checkbox
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import threading

# ================= Robot Params & Kinematics =================
L1, L2, L3 = 0.6, 0.6, 0.6
D6 = 0.1
J4_OFFSET_Y = 0

def get_transform(theta, d, a, alpha):
    return np.array([
        [np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
        [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
        [0,              np.sin(alpha),                np.cos(alpha),               d],
        [0,              0,                            0,                           1]
    ])

def forward_kinematics_visual(q, l1, l2, l3, d6, offset_y=0, base_y=0.0):
    q1, q2, q3, q4, q5, q6 = q
    a1 = 0
    gamma = np.arctan2(offset_y, l3)
    l3_eff = np.sqrt(l3**2 + offset_y**2)
    dh_params = [
        [q1, l1, a1, -np.pi/2], 
        [q2, 0, l2, 0], 
        [q3 + np.pi/2 - gamma, 0, 0, np.pi/2], 
        [q4, l3_eff, 0, -np.pi/2], 
        [q5+gamma, 0, 0, np.pi/2], 
        [q6, d6, 0, 0]  
    ]
    T = np.eye(4); T[1, 3] = base_y 
    T_list = [T.copy()]
    for params in dh_params:
        T = T @ get_transform(*params)
        T_list.append(T.copy())
    return T_list

# ================= ROS2 Node (Listener Only) =================
class JointStateVisualizerNode(Node):
    def __init__(self):
        super().__init__('robot_visualizer_node')
        self.subscriber = self.create_subscription(
            JointState, 
            'joint_states', 
            self.joint_state_callback, 
            10
        )
        # ตัวแปรสำหรับเก็บค่า Joint ล่าสุด
        self.current_track_y = 0.0
        self.current_joints = [0.0] * 6

    def joint_state_callback(self, msg):
        try:
            joint_dict = dict(zip(msg.name, msg.position))
            
            if 'track_y' in joint_dict:
                self.current_track_y = joint_dict['track_y']
            
            for i in range(6):
                j_name = f'joint_{i+1}'
                if j_name in joint_dict:
                    self.current_joints[i] = joint_dict[j_name]
                    
        except Exception as e:
            self.get_logger().error(f"Error parsing joint states: {e}")

# ================= Plot & Animation Setup =================
rclpy.init()
node = JointStateVisualizerNode()
ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
ros_thread.start()

# --- ตั้งค่าสถานะการมองเห็นของแกน (เปิดไว้ทั้งหมดเป็น Default) ---
labels_axes = ["Base", "J1", "J2", "J3", "J4", "J5", "J6"]
axes_visibility = {label: True for label in labels_axes}

fig = plt.figure(figsize=(11, 8))
# ขยับกราฟ 3 มิติไปทางขวาเล็กน้อย เพื่อเหลือพื้นที่ด้านซ้ายให้ปุ่ม
plt.subplots_adjust(left=0.25) 
ax = fig.add_subplot(111, projection='3d')

def draw_axes(ax, T, length=0.08, label=""):
    origin = T[:3, 3]
    ax.quiver(*origin, *(T[:3, 0] * length), color='r', linewidth=1.5)
    ax.quiver(*origin, *(T[:3, 1] * length), color='g', linewidth=1.5)
    ax.quiver(*origin, *(T[:3, 2] * length), color='b', linewidth=3.0) 
    if label:
        offset_z = -0.04 if label in ["J3", "J5"] else 0.02
        ax.text(origin[0], origin[1], origin[2] + offset_z, label, fontsize=9, fontweight='bold')

def update_plot(frame):
    ax.cla()
    
    # ดึงค่าปัจจุบันจาก ROS Node
    track_y = node.current_track_y
    joints = node.current_joints

    # คำนวณ Forward Kinematics
    T_list = forward_kinematics_visual(joints, L1, L2, L3, D6, J4_OFFSET_Y, track_y)
    pts = np.array([T[:3, 3] for T in T_list])
    
    ee_pos = pts[-1] 

    # วาด Linear Track และแขนหุ่นยนต์
    ax.plot([0, 0], [-1.0, 1.0], [0, 0], '--', color='black', linewidth=3, alpha=0.5, label="Linear Track")
    ax.plot(pts[:,0], pts[:,1], pts[:,2], '-o', color='#34495e', linewidth=4, alpha=0.8)
    ax.scatter(ee_pos[0], ee_pos[1], ee_pos[2], color='purple', s=50, label="End Effector")

    # วาดแกน (Frames) โดยเช็คจากสถานะปุ่ม Checkbox
    for i, T in enumerate(T_list):
        label = labels_axes[i]
        if axes_visibility[label]: # วาดเฉพาะแกนที่ตั้งเป็น True
            draw_axes(ax, T, length=0.08, label=label)

    # กล่องข้อความแสดงค่า
    angle_text = f"Track Y: {track_y:.3f} m\n\nCurrent Joints:\n"
    for i in range(6): 
        angle_text += f"q{i+1}: {joints[i]:.2f} rad ({np.degrees(joints[i]):.1f}°)\n"
        
    angle_text += f"\nEnd Effector (XYZ):\n"
    angle_text += f"X: {ee_pos[0]:.2f} m\n"
    angle_text += f"Y: {ee_pos[1]:.2f} m\n"
    angle_text += f"Z: {ee_pos[2]:.2f} m\n"

    # ขยับตำแหน่ง Text ไปทางซ้ายของแกน 3 มิติ เพื่อไม่ให้ทับกับโมเดล
    ax.text2D(-0.25, 0.50, angle_text, transform=ax.transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

    # ตั้งค่าขอบเขตและมุมมอง
    ax.set_xlim([-1.0, 1.0])
    ax.set_ylim([-1.0, 1.0])
    ax.set_zlim([0, 1.5])
    ax.set_xlabel('Global X')
    ax.set_ylabel('Global Y')
    ax.set_zlabel('Global Z')
    ax.legend(loc="upper right")
    ax.set_title("7-Axis Robot Live Visualizer")

# --- สร้างปุ่ม Checkbox ทางมุมซ้ายล่าง ---
# พิกัด [left, bottom, width, height] เทียบกับขนาดหน้าต่างทั้งหมด
ax_checkbox = plt.axes([0.02, 0.1, 0.12, 0.35]) 
chk = CheckButtons(ax_checkbox, labels_axes, [True]*len(labels_axes))

# ฟังก์ชันอัปเดตสถานะเมื่อมีการกดคลิกที่ปุ่ม
def toggle_axes(label):
    axes_visibility[label] = not axes_visibility[label]

# ผูกปุ่มเข้ากับฟังก์ชัน
chk.on_clicked(toggle_axes)

# ตั้งเวลาให้รูปอัปเดตตัวเองอัตโนมัติ
ani = FuncAnimation(fig, update_plot, interval=50, cache_frame_data=False)

plt.show()

# เมื่อปิดหน้าต่างกราฟ ให้ปิด ROS Node ด้วย
rclpy.shutdown()
ros_thread.join()