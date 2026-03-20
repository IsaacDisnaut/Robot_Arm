import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons, RadioButtons, Button
from mpl_toolkits.mplot3d import Axes3D
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import threading
import time
import json
import os

# ================= Configuration =================
STATE_FILE = 'tasks/data.txt'

# ================= ROS2 Node =================
class JointControlNode(Node):
    def __init__(self):
        super().__init__('ik_joint_control_gui')
        self.publisher = self.create_publisher(JointState, 'joint_states', 10)
        self.subscriber = self.create_subscription(JointState, 'joint_states', self.joint_state_callback, 10)
        
        self.initial_base_y = 0.0
        self.initial_joints = [0.0] * 6
        self.state_received_event = threading.Event()
        self.ignore_sub = False 

    def joint_state_callback(self, msg):
        if self.ignore_sub: return
        try:
            joint_dict = dict(zip(msg.name, msg.position))
            self.initial_base_y = joint_dict.get('slider_joint', 0.0)
            self.initial_joints = [
                joint_dict.get('joint_1', 0.0), joint_dict.get('joint_2', 0.0),
                joint_dict.get('joint_3', 0.0), joint_dict.get('joint_4', 0.0),
                joint_dict.get('joint_5', 0.0), joint_dict.get('joint_6', 0.0)
            ]
            self.ignore_sub = True 
            self.state_received_event.set() 
        except Exception as e:
            self.get_logger().error(f"Error parsing joint states: {e}")

    def publish_joints(self, joints, base_y):
        joint_msg = JointState()
        joint_msg.header.stamp = self.get_clock().now().to_msg()
        joint_msg.name = [
            'slider_joint', 'joint_1', 'joint_2', 'joint_3',
            'joint_4', 'joint_5', 'joint_6'
        ]
        joint_msg.position = [float(base_y)] + [float(q) for q in joints]
        self.publisher.publish(joint_msg)

# ================= Robot Params & Kinematics =================
L1, L2, L3 = 0.28787, 0.26096, 0.26136
D6 = 0.07074
J4_OFFSET_Y = 0.02175 

def rpy_to_matrix(roll, pitch, yaw):
    Rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx

def matrix_to_rpy(R):
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[0, 0]**2 + R[1, 0]**2))
    roll = np.arctan2(R[2, 1], R[2, 2])
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return roll, pitch, yaw

def get_transform(theta, d, a, alpha):
    return np.array([
        [np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
        [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
        [0,              np.sin(alpha),                np.cos(alpha),               d],
        [0,              0,                            0,                           1]
    ])

def forward_kinematics_clean(q, l1, l2, l3, d6):
    a1 = 0.020885
    dh_params = [
        [q[0], l1, a1, -np.pi/2], [q[1], 0, l2, 0], [q[2] + np.pi/2, 0, 0, np.pi/2], 
        [q[3], l3, 0, -np.pi/2], [q[4], 0, 0, np.pi/2], [q[5], d6, 0, 0]  
    ]
    T = np.eye(4)
    for params in dh_params: T = T @ get_transform(*params)
    return T

def forward_kinematics_visual(q, l1, l2, l3, d6, offset_y, base_y=0.0):
    q1, q2, q3, q4, q5, q6 = q
    a1 = 0.020885
    gamma = np.arctan2(offset_y, l3)
    l3_eff = np.sqrt(l3**2 + offset_y**2)
    dh_params = [
        [q1, l1, a1, -np.pi/2], [q2, 0, l2, 0], [q3 + np.pi/2 - gamma, 0, 0, np.pi/2], 
        [q4, l3_eff, 0, -np.pi/2], [q5+gamma, 0, 0, np.pi/2], [q6, d6, 0, 0]  
    ]
    T = np.eye(4); T[1, 3] = base_y 
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
    dh03 = [[q1, l1, a1, -np.pi/2], [q2, 0, l2, 0], [q3 + np.pi/2, 0, 0, np.pi/2]]
    for params in dh03: T0 = T0 @ get_transform(*params)
    R03 = T0[:3, :3]
    R36 = R03.T @ target_orient
    q5 = np.arctan2(np.sqrt(R36[0,2]**2 + R36[1,2]**2), R36[2,2])

    if np.abs(R36[2, 2]) > 0.9999: 
        q4 = 0; q6 = np.arctan2(R36[1,0], R36[0,0])
    else:
        q4 = np.arctan2(R36[1,2], R36[0,2]); q6 = np.arctan2(R36[2,1], -R36[2,0])

    return np.array([q1, q2, q3, q4, q5, q6]), reachable


# ================= ROS Initialization & Fetch State =================
rclpy.init()
node = JointControlNode()
ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
ros_thread.start()

print("Waiting for initial joint states from ROS (timeout in 1.5s)...")
if node.state_received_event.wait(timeout=1.5):
    # กรณีที่ 1: มี Node อื่น (เช่น ตัวหุ่นยนต์จริง) Publish อยู่
    print("Received current joint states from ROS!")
    init_base_y = node.initial_base_y
    init_q = node.initial_joints
else:
    # กรณีที่ 2: ไม่มีใคร Publish เลย (เพิ่งเปิดโปรแกรมใหม่) -> ไปโหลดจากไฟล์
    print("No active ROS publisher found. Checking local save file...")
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'r') as f:
                data = json.load(f)
                init_base_y = data.get('base_y', 0.0)
                init_q = data.get('joints', [0.0]*6)
            print(f"Loaded previous state from '{STATE_FILE}'.")
        except Exception as e:
            print(f"Failed to read '{STATE_FILE}'. Using default values. Error: {e}")
            init_base_y = 0.0
            init_q = [0.0] * 6
    else:
        print("No previous save file found. Using default values (0.0).")
        init_base_y = 0.0
        init_q = [0.0] * 6

# คำนวณ IK ย้อนกลับเพื่อหาค่า X,Y,Z,R,P,Y เริ่มต้นให้ฝั่ง IK Sliders
T_init = forward_kinematics_clean(init_q, L1, L2, L3, D6)
init_pos = T_init[:3, 3]
init_r, init_p, init_y = matrix_to_rpy(T_init[:3, :3])

init_ik = [
    init_pos[0], init_pos[1], init_pos[2],
    np.degrees(init_r), np.degrees(init_p), np.degrees(init_y)
]

# ================= Plot & UI Setup =================
fig = plt.figure(figsize=(13, 9))
plt.subplots_adjust(left=0.25, bottom=0.35)
ax = fig.add_subplot(111, projection='3d')

global_ui_elements = [] 

def make_slider_with_btns(y_pos, label, vmin, vmax, vinit, step, color, wrap=False):
    ax_s = plt.axes([0.30, y_pos, 0.45, 0.02])
    ax_m = plt.axes([0.76, y_pos, 0.03, 0.02])
    ax_p = plt.axes([0.80, y_pos, 0.03, 0.02])
    
    s = Slider(ax_s, label, vmin, vmax, valinit=vinit, color=color)
    btn_m = Button(ax_m, '-')
    btn_p = Button(ax_p, '+')

    timer = ax_s.figure.canvas.new_timer(interval=100)
    timer_inc = [0] 

    def adjust_val(inc):
        new_val = s.val + inc
        if wrap:
            new_val = vmin + (new_val - vmin) % (vmax - vmin)
        else:
            new_val = max(vmin, min(vmax, new_val))
        s.set_val(new_val)

    def on_timer():
        adjust_val(timer_inc[0])

    timer.add_callback(on_timer)

    def on_press(event):
        if event.button != 1: return
        if event.inaxes == ax_m and ax_m.get_visible():
            timer_inc[0] = -step; adjust_val(-step); timer.start()     
        elif event.inaxes == ax_p and ax_p.get_visible():
            timer_inc[0] = step; adjust_val(step); timer.start()

    def on_release(event):
        timer.stop()

    ax_s.figure.canvas.mpl_connect('button_press_event', on_press)
    ax_s.figure.canvas.mpl_connect('button_release_event', on_release)
    
    global_ui_elements.extend([btn_m, btn_p, timer]) 
    return s, btn_m, btn_p

# --- สร้าง Base Slider ---
s_base_y, b_base_m, b_base_p = make_slider_with_btns(0.28, 'Base Track Y', -1.0, 1.0, init_base_y, 0.05, 'orange', wrap=False)

# --- 1. IK Sliders ---
ik_labels = ['Local X', 'Local Y', 'Local Z', 'Roll (deg)', 'Pitch (deg)', 'Yaw (deg)']
ik_bounds = [(-1.0, 1.0), (-1.0, 1.0), (0.0, 1.5), (-180, 180), (-180, 180), (-180, 180)]
ik_steps = [0.05, 0.05, 0.05, 5.0, 5.0, 5.0] 
ik_wraps = [False, False, False, True, True, True] 

ik_sliders, ik_btns = [], []
for i in range(6):
    s, bm, bp = make_slider_with_btns(
        0.24 - i*0.035, ik_labels[i], ik_bounds[i][0], ik_bounds[i][1], init_ik[i], ik_steps[i], 'skyblue', wrap=ik_wraps[i]
    )
    ik_sliders.append(s)
    ik_btns.extend([bm, bp])

# --- 2. Joint Sliders ---
j_sliders, j_btns = [], []
for i in range(6):
    init_q_deg = np.degrees(init_q[i])
    s, bm, bp = make_slider_with_btns(
        0.24 - i*0.035, f'q{i+1} (deg)', -180, 180, init_q_deg, 5.0, 'lightgreen', wrap=True
    )
    s.ax.set_visible(False); bm.ax.set_visible(False); bp.ax.set_visible(False)
    j_sliders.append(s)
    j_btns.extend([bm, bp])

# --- UI สลับโหมดและ Checkbox ---
ax_radio = plt.axes([0.02, 0.75, 0.18, 0.12], facecolor='#f0f0f0')
radio_mode = RadioButtons(ax_radio, ('IK Mode\n(XYZ RPY)', 'Joint Mode\n(q1 - q6)'))
current_mode = 'IK Mode\n(XYZ RPY)'

ax_check = plt.axes([0.02, 0.45, 0.12, 0.25], facecolor='#f0f0f0')
labels_axes = ["Base", "J1", "J2", "J3", "J4", "J5", "J6"]
axes_visibility = [True] * 7 
check = CheckButtons(ax_check, labels_axes, axes_visibility)

def toggle_axes(label):
    idx = labels_axes.index(label)
    axes_visibility[idx] = not axes_visibility[idx]
    update(None)
check.on_clicked(toggle_axes)

def switch_mode(label):
    global current_mode
    current_mode = label
    is_ik = 'IK Mode' in label
    
    for s in ik_sliders: s.ax.set_visible(is_ik)
    for b in ik_btns: b.ax.set_visible(is_ik)
        
    for s in j_sliders: s.ax.set_visible(not is_ik)
    for b in j_btns: b.ax.set_visible(not is_ik)
        
    fig.canvas.draw_idle()
    update(None)

radio_mode.on_clicked(switch_mode)

def draw_axes(ax, T, length=0.08, label=""):
    origin = T[:3, 3]
    ax.quiver(*origin, *(T[:3, 0] * length), color='r', linewidth=1.5)
    ax.quiver(*origin, *(T[:3, 1] * length), color='g', linewidth=1.5)
    ax.quiver(*origin, *(T[:3, 2] * length), color='b', linewidth=3.0) 
    if label:
        offset_z = -0.04 if label in ["J3", "J5"] else 0.02
        ax.text(origin[0], origin[1], origin[2] + offset_z, label, fontsize=9, fontweight='bold')

def update(val):
    ax.cla()
    base_y = s_base_y.val
    reachable = True

    if 'IK Mode' in current_mode:
        local_target_pos = np.array([ik_sliders[0].val, ik_sliders[1].val, ik_sliders[2].val])
        r, p, y = np.radians([ik_sliders[3].val, ik_sliders[4].val, ik_sliders[5].val])
        target_orient = rpy_to_matrix(r, p, y)

        joints, reachable = inverse_kinematics_6dof(local_target_pos, target_orient, L1, L2, L3, D6)
        
        for i in range(6):
            j_sliders[i].eventson = False
            j_sliders[i].set_val(np.degrees(joints[i]))
            j_sliders[i].eventson = True
            
    else:
        joints = np.radians([s.val for s in j_sliders])
        T_clean = forward_kinematics_clean(joints, L1, L2, L3, D6)
        local_pos = T_clean[:3, 3]
        r, p, y = matrix_to_rpy(T_clean[:3, :3])
        
        ik_vals = [local_pos[0], local_pos[1], local_pos[2], np.degrees(r), np.degrees(p), np.degrees(y)]
        
        for i in range(6):
            ik_sliders[i].eventson = False
            ik_sliders[i].set_val(ik_vals[i])
            ik_sliders[i].eventson = True

    T_list = forward_kinematics_visual(joints, L1, L2, L3, D6, J4_OFFSET_Y, base_y)
    pts = np.array([T[:3, 3] for T in T_list])

    ax.plot([0, 0], [-1.0, 1.0], [0, 0], '--', color='black', linewidth=3, alpha=0.5, label="Linear Track")
    ax.plot(pts[:,0], pts[:,1], pts[:,2], '-o', color='#34495e', linewidth=4, alpha=0.8)
    ax.scatter(pts[-1,0], pts[-1,1], pts[-1,2], color='purple', s=50, label="End Effector")

    if 'IK Mode' in current_mode:
        global_target_pos = local_target_pos + np.array([0, base_y, 0])
        target_color = 'green' if reachable else 'red'
        ax.scatter(global_target_pos[0], global_target_pos[1], global_target_pos[2], color=target_color, s=100, label="Target (Local)")
        if not reachable:
            ax.text2D(0.05, 0.95, "Target is UNREACHABLE!", transform=ax.transAxes, color='red', fontsize=14, fontweight='bold')

    for i, T in enumerate(T_list):
        if axes_visibility[i]: draw_axes(ax, T, length=0.08, label=labels_axes[i])

    q_deg = np.degrees(joints)
    angle_text = f"Mode: {current_mode.split()[0]}\nTrack Y: {base_y:.2f} m\n\nJoints (to ROS):\n"
    for i in range(6): angle_text += f"q{i+1}: {joints[i]:.2f} rad ({q_deg[i]:.1f}°)\n"
    ax.text2D(0.02, 0.70, angle_text, transform=ax.transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

    node.publish_joints(joints, base_y)

    ax.set_xlim([-1.0, 1.0]); ax.set_ylim([-1.0, 1.0]); ax.set_zlim([0, 1.5])
    ax.set_xlabel('Global X'); ax.set_ylabel('Global Y'); ax.set_zlabel('Global Z')
    ax.legend(loc="upper right")
    ax.set_title("7-Axis System Control: IK vs Joint Mode")

    fig.canvas.draw_idle()

# 🔥 ระบบเซฟไฟล์ตอนปิดหน้าต่าง
def on_close(event):
    state = {
        'base_y': s_base_y.val,
        'joints': [np.radians(s.val) for s in j_sliders]
    }
    try:
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f)
        print(f"✅ Saved current state to '{STATE_FILE}' before exit.")
    except Exception as e:
        print(f"❌ Failed to save state: {e}")

fig.canvas.mpl_connect('close_event', on_close)

s_base_y.on_changed(update)
for s in ik_sliders + j_sliders:
    s.on_changed(update)

update(None)
plt.show()

rclpy.shutdown()