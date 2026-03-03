import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons
from mpl_toolkits.mplot3d import Axes3D
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import threading

# ================= ROS2 Node =================
class JointPublisher(Node):
    def __init__(self):
        super().__init__('ik_joint_publisher')
        self.publisher = self.create_publisher(JointState, 'joint_states', 10)

    def publish_joints(self, joints, base_y):
        joint_msg = JointState()
        joint_msg.header.stamp = self.get_clock().now().to_msg()
        # 🔥 เพิ่มชื่อแกนรางเลื่อน (track_y) เข้าไปเป็นแกนที่ 7
        joint_msg.name = [
            'track_y', 'joint_1', 'joint_2', 'joint_3',
            'joint_4', 'joint_5', 'joint_6'
        ]
        joint_msg.position = [float(base_y)] + [float(q) for q in joints]
        self.publisher.publish(joint_msg)

# ================= Robot Params =================
L1, L2, L3 = 0.28787, 0.26096, 0.26136
D6 = 0.07074

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

def inverse_kinematics_6dof(target_pos, target_orient, l1, l2, l3, d6):
    xc, yc, zc = target_pos - d6 * target_orient[:, 2]

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
        [q1, l1, 0, -np.pi/2],
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
plt.subplots_adjust(left=0.2, bottom=0.40) # ขยายพื้นที่ด้านล่างเพื่อใส่ Slider รางเลื่อน
ax = fig.add_subplot(111, projection='3d')

init_x, init_y, init_z = (L2 + L3 + D6), 0.0, L1
init_roll, init_pitch, init_yaw = 0, 90, 0
init_base_y = 0.0

# 🔥 เพิ่ม Slider รางเลื่อนฐานหุ่นยนต์ (Base Y Track)
ax_base_y = plt.axes([0.25, 0.33, 0.65, 0.03])
ax_x      = plt.axes([0.25, 0.29, 0.65, 0.03])
ax_y      = plt.axes([0.25, 0.25, 0.65, 0.03])
ax_z      = plt.axes([0.25, 0.21, 0.65, 0.03])
ax_roll   = plt.axes([0.25, 0.17, 0.65, 0.03])
ax_pitch  = plt.axes([0.25, 0.13, 0.65, 0.03])
ax_yaw    = plt.axes([0.25, 0.09, 0.65, 0.03])

# Slider สีพิเศษเพื่อให้เห็นชัดว่าเป็นรางเลื่อน
s_base_y = Slider(ax_base_y, 'Base Track Y', -1.0, 1.0, valinit=init_base_y, color='orange')
s_x = Slider(ax_x, 'Target X', -1.0, 1.0, valinit=init_x)
s_y = Slider(ax_y, 'Target Y', -1.0, 1.0, valinit=init_y)
s_z = Slider(ax_z, 'Target Z', 0.0, 1.5, valinit=init_z)
s_roll = Slider(ax_roll, 'Roll (deg)', -180, 180, valinit=init_roll)
s_pitch = Slider(ax_pitch, 'Pitch (deg)', -180, 180, valinit=init_pitch)
s_yaw = Slider(ax_yaw, 'Yaw (deg)', -180, 180, valinit=init_yaw)

sliders = [s_base_y, s_x, s_y, s_z, s_roll, s_pitch, s_yaw]

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
    target_pos = np.array([s_x.val, s_y.val, s_z.val])
    r, p, y = np.radians(s_roll.val), np.radians(s_pitch.val), np.radians(s_yaw.val)
    target_orient = rpy_to_matrix(r, p, y)

    # 🔥 หักลบระยะรางเลื่อน เพื่อให้ IK คำนวณเสมือนว่าเป้าหมายเคลื่อนเข้ามาหาฐานแทน
    local_target_pos = target_pos - np.array([0, base_y, 0])

    joints, reachable = inverse_kinematics_6dof(local_target_pos, target_orient, L1, L2, L3, D6)
    
    T_list = forward_kinematics_matrices(joints, L1, L2, L3, D6, base_y)
    pts = np.array([T[:3, 3] for T in T_list])

    # วาดรางเลื่อน (Linear Track) เส้นประสีดำ
    ax.plot([0, 0], [-1.0, 1.0], [0, 0], '--', color='black', linewidth=3, alpha=0.5, label="Linear Track (Y)")
    
    # วาดเส้นแขนกล
    ax.plot(pts[:,0], pts[:,1], pts[:,2], '-o', color='#34495e', linewidth=4, alpha=0.8)

    target_color = 'green' if reachable else 'red'
    ax.scatter(target_pos[0], target_pos[1], target_pos[2], color=target_color, s=100, label="Target Input")
    ax.scatter(pts[-1,0], pts[-1,1], pts[-1,2], color='purple', s=50, label="Real IK Output")

    for i, T in enumerate(T_list):
        if axes_visibility[i]:
            draw_axes(ax, T, length=0.08, label=labels[i])

    info_text = "Target is UNREACHABLE!" if not reachable else ""
    ax.text2D(0.05, 0.95, info_text, transform=ax.transAxes, color='red', fontsize=14, fontweight='bold')

    q_deg = np.degrees(joints)
    angle_text = f"Track Y: {base_y:.2f} m\n\nCalculated Joints:\n"
    for i in range(6):
        angle_text += f"q{i+1}: {joints[i]:.2f} rad ({q_deg[i]:.1f}°)\n"
    ax.text2D(0.02, 0.65, angle_text, transform=ax.transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

    # ส่งค่าเข้า ROS 2 (รวมแกน Track Y ด้วย)
    node.publish_joints(joints, base_y)

    ax.set_xlim([-1.0, 1.0]); ax.set_ylim([-1.0, 1.0]); ax.set_zlim([0, 1.5])
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.legend(loc="upper right")
    ax.set_title("7-DOF IK System: 6-Axis Robot on a Linear Track")

    fig.canvas.draw_idle()

for s in sliders:
    s.on_changed(update)

update(None)
plt.show()

rclpy.shutdown()