#!/usr/bin/env python3
"""
EE Pose Slider GUI
──────────────────
Sliders for X / Y / Z / Roll / Pitch / Yaw.
On every slider change:
  1. Solve IK  →  joint angles
  2. Publish JointState to /motor

Run:
  python3 ee_pose_gui.py
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

# ──────────────────────────────────────────────
#  Robot constants  (same as arm_controller.py)
# ──────────────────────────────────────────────
L1, L2, L3 = 0.28787, 0.26096, 0.26136
D6          = 0.07074
J4_OFFSET_Y = 0.02175
A1          = 0.020885

Q_MIN = np.radians([-180.0, -180.0, -120.0, -180.0, -100.0, -360.0])
Q_MAX = np.radians([ 180.0,   30.0,  150.0,  180.0,  100.0,  360.0])

SLIDER_MIN = -1.0
SLIDER_MAX =  1.0

# ──────────────────────────────────────────────
#  Kinematics helpers
# ──────────────────────────────────────────────
def get_transform(theta, d, a, alpha):
    c, s  = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ c, -s*ca,  s*sa, a*c],
        [ s,  c*ca, -c*sa, a*s],
        [ 0,    sa,    ca,   d],
        [ 0,     0,     0,   1],
    ])

def forward_kinematics(q, base_y=0.0):
    q1, q2, q3, q4, q5, q6 = q
    dh = [
        [q1,           L1, A1,       -np.pi/2],
        [q2,            0, L2,        0      ],
        [q3 + np.pi/2,  0,  0,        np.pi/2],
        [q4,           L3,  0,       -np.pi/2],
        [q5,            0,  0,        np.pi/2],
        [q6,           D6,  0,        0      ],
    ]
    T = np.eye(4)
    T[1, 3] = base_y
    for p in dh:
        T = T @ get_transform(*p)
    return T

def inverse_kinematics(target_pos, target_orient, base_y=0.0):
    """Returns (q[6], reachable:bool)"""
    local_pos = target_pos - np.array([0.0, base_y, 0.0])
    xc, yc, zc = local_pos - D6 * target_orient[:, 2]

    q1 = np.arctan2(yc, xc)
    r  = np.sqrt(xc**2 + yc**2)
    s  = zc - L1
    D_sq = r**2 + s**2

    cos_q3 = (D_sq - L2**2 - L3**2) / (2 * L2 * L3)
    reachable = True
    if cos_q3 > 1.0 or cos_q3 < -1.0:
        reachable = False
        cos_q3 = np.clip(cos_q3, -1.0, 1.0)

    sin_q3 = np.sqrt(max(0.0, 1 - cos_q3**2))
    q3     = np.arctan2(sin_q3, cos_q3)
    beta   = np.arctan2(L3 * np.sin(q3), L2 + L3 * np.cos(q3))
    q2     = np.arctan2(-s, r) - beta

    # R03
    T0 = np.eye(4)
    for p in [
        [q1,           L1, A1, -np.pi/2],
        [q2,            0, L2,  0      ],
        [q3 + np.pi/2,  0,  0,  np.pi/2],
    ]:
        T0 = T0 @ get_transform(*p)
    R03 = T0[:3, :3]

    R36 = R03.T @ target_orient
    q5  = np.arctan2(np.sqrt(R36[0,2]**2 + R36[1,2]**2), R36[2,2])

    if np.abs(R36[2, 2]) > 0.9999:   # singularity
        q4 = 0.0
        q6 = np.arctan2(R36[1,0], R36[0,0])
    else:
        q4 = np.arctan2(R36[1,2], R36[0,2])
        q6 = np.arctan2(R36[2,1], -R36[2,0])

    return np.array([q1, q2, q3, q4, q5, q6]), reachable

def rpy_to_matrix(roll, pitch, yaw):
    Rx = np.array([[1,0,0],[0,np.cos(roll),-np.sin(roll)],[0,np.sin(roll),np.cos(roll)]])
    Ry = np.array([[np.cos(pitch),0,np.sin(pitch)],[0,1,0],[-np.sin(pitch),0,np.cos(pitch)]])
    Rz = np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1]])
    return Rz @ Ry @ Rx

def matrix_to_rpy(R):
    pitch = np.arctan2(-R[2,0], np.sqrt(R[0,0]**2 + R[1,0]**2))
    if np.abs(np.abs(pitch) - np.pi/2) < 1e-6:
        yaw  = 0.0
        roll = np.arctan2(R[0,1], R[0,2]) * (1 if pitch > 0 else -1)
    else:
        yaw  = np.arctan2(R[1,0], R[0,0])
        roll = np.arctan2(R[2,1], R[2,2])
    return roll, pitch, yaw

# ──────────────────────────────────────────────
#  ROS2 Node
# ──────────────────────────────────────────────
class GUINode(Node):
    def __init__(self):
        super().__init__('ee_pose_gui')
        self.publisher = self.create_publisher(JointState, '/motor', 10)
        self.sub = self.create_subscription(
            JointState, '/joint_states', self._joint_cb, 10)

        self.current_joints = [0.0] * 6
        self.current_slider = 0.0
        self.received       = False

    def _joint_cb(self, msg):
        if len(msg.position) >= 7:
            self.current_slider = msg.position[0]
            self.current_joints = list(msg.position[1:7])
            self.received = True

    def publish(self, joints, slider=0.0):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name     = ['slider_joint','joint_1','joint_2','joint_3',
                        'joint_4','joint_5','joint_6']
        msg.position = [float(slider)] + [float(q) for q in joints]
        self.publisher.publish(msg)

# ──────────────────────────────────────────────
#  GUI
# ──────────────────────────────────────────────
class App:
    # Slider ranges
    RANGES = {
        'X (m)':     (-0.8,  0.8,  0.001),
        'Y (m)':     (-0.8,  0.8,  0.001),
        'Z (m)':     ( 0.0,  1.2,  0.001),
        'Roll (°)':  (-180, 180,   0.1  ),
        'Pitch (°)': (-180, 180,   0.1  ),
        'Yaw (°)':   (-180, 180,   0.1  ),
    }
    KEYS = list(RANGES.keys())

    BG      = '#0f1117'
    PANEL   = '#1a1d27'
    ACCENT  = '#00e5ff'
    ACCENT2 = '#ff4081'
    TEXT    = '#e8eaf6'
    MUTED   = '#555870'
    GREEN   = '#69ff6e'
    RED     = '#ff5252'

    def __init__(self, ros_node: GUINode):
        self.node = ros_node
        self.last_q = list(ros_node.current_joints)

        self.root = tk.Tk()
        self.root.title('EE Pose Control')
        self.root.configure(bg=self.BG)
        self.root.resizable(False, False)

        self._build_ui()
        self._wait_for_joints()

    # ── UI construction ────────────────────────
    def _build_ui(self):
        root = self.root

        # Header
        hdr = tk.Frame(root, bg=self.BG)
        hdr.pack(fill='x', padx=20, pady=(18, 4))
        tk.Label(hdr, text='END-EFFECTOR POSE CONTROL',
                 font=('Courier New', 13, 'bold'),
                 fg=self.ACCENT, bg=self.BG).pack(side='left')
        self.status_dot = tk.Label(hdr, text='●', font=('Courier New', 14),
                                   fg=self.RED, bg=self.BG)
        self.status_dot.pack(side='right')
        self.status_lbl = tk.Label(hdr, text='waiting for /joint_states',
                                   font=('Courier New', 9),
                                   fg=self.MUTED, bg=self.BG)
        self.status_lbl.pack(side='right', padx=(0,6))

        sep = tk.Frame(root, bg=self.ACCENT, height=1)
        sep.pack(fill='x', padx=20, pady=(0,12))

        # Notebook tabs
        style = ttk.Style()
        style.theme_use('default')
        style.configure('Dark.TNotebook', background=self.BG, borderwidth=0)
        style.configure('Dark.TNotebook.Tab',
                        background=self.PANEL, foreground=self.MUTED,
                        font=('Courier New', 10, 'bold'),
                        padding=[16, 6])
        style.map('Dark.TNotebook.Tab',
                  background=[('selected', self.BG)],
                  foreground=[('selected', self.ACCENT)])

        nb = ttk.Notebook(root, style='Dark.TNotebook')
        nb.pack(padx=20, pady=0, fill='both')

        self.sliders = {}
        self.value_vars = {}
        self.entry_vars = {}

        # Position tab
        pos_frame = tk.Frame(nb, bg=self.BG)
        nb.add(pos_frame, text='  POSITION  ')
        for key in self.KEYS[:3]:
            self._add_slider_row(pos_frame, key)

        # Orientation tab
        rot_frame = tk.Frame(nb, bg=self.BG)
        nb.add(rot_frame, text='  ORIENTATION  ')
        for key in self.KEYS[3:]:
            self._add_slider_row(rot_frame, key)

        # Joint readout panel
        jnt_frame = tk.Frame(root, bg=self.PANEL, bd=0)
        jnt_frame.pack(padx=20, pady=(14, 0), fill='x')

        tk.Label(jnt_frame, text=' JOINT SOLUTION',
                 font=('Courier New', 9, 'bold'),
                 fg=self.MUTED, bg=self.PANEL).grid(
                     row=0, column=0, columnspan=6, sticky='w', padx=8, pady=(6,2))

        self.joint_vars = []
        for i in range(6):
            v = tk.StringVar(value='—')
            self.joint_vars.append(v)
            col_frame = tk.Frame(jnt_frame, bg=self.PANEL)
            col_frame.grid(row=1, column=i, padx=8, pady=(0,8))
            tk.Label(col_frame, text=f'J{i+1}',
                     font=('Courier New', 8), fg=self.MUTED,
                     bg=self.PANEL).pack()
            tk.Label(col_frame, textvariable=v,
                     font=('Courier New', 10, 'bold'),
                     fg=self.ACCENT2, bg=self.PANEL, width=7).pack()

        # IK status
        self.ik_var = tk.StringVar(value='')
        tk.Label(jnt_frame, textvariable=self.ik_var,
                 font=('Courier New', 9),
                 fg=self.RED, bg=self.PANEL).grid(
                     row=2, column=0, columnspan=6, sticky='w', padx=8, pady=(0,6))

        # Buttons
        btn_frame = tk.Frame(root, bg=self.BG)
        btn_frame.pack(padx=20, pady=14, fill='x')

        tk.Button(btn_frame, text='SYNC FROM ROBOT',
                  font=('Courier New', 10, 'bold'),
                  bg=self.PANEL, fg=self.ACCENT,
                  activebackground=self.ACCENT, activeforeground=self.BG,
                  bd=0, padx=16, pady=8, cursor='hand2',
                  command=self._sync_from_robot).pack(side='left', padx=(0,10))

        tk.Button(btn_frame, text='PUBLISH',
                  font=('Courier New', 10, 'bold'),
                  bg=self.ACCENT, fg=self.BG,
                  activebackground='#00bcd4', activeforeground=self.BG,
                  bd=0, padx=24, pady=8, cursor='hand2',
                  command=self._publish_now).pack(side='left')

        # Footer
        tk.Frame(root, bg=self.MUTED, height=1).pack(fill='x', padx=20, pady=(0,4))
        tk.Label(root, text='publishes to /motor on every slider change',
                 font=('Courier New', 8), fg=self.MUTED, bg=self.BG).pack(pady=(0,10))

    def _add_slider_row(self, parent, key):
        lo, hi, res = self.RANGES[key]
        frame = tk.Frame(parent, bg=self.BG)
        frame.pack(fill='x', padx=16, pady=6)

        # Label
        tk.Label(frame, text=key, width=11, anchor='w',
                 font=('Courier New', 10, 'bold'),
                 fg=self.TEXT, bg=self.BG).pack(side='left')

        # Value display
        val_var = tk.StringVar(value='0.000')
        self.value_vars[key] = val_var
        tk.Label(frame, textvariable=val_var, width=8, anchor='e',
                 font=('Courier New', 10),
                 fg=self.ACCENT, bg=self.BG).pack(side='right', padx=(6,0))

        # Entry box
        ent_var = tk.StringVar()
        self.entry_vars[key] = ent_var
        ent = tk.Entry(frame, textvariable=ent_var, width=7,
                       font=('Courier New', 9),
                       bg=self.PANEL, fg=self.TEXT,
                       insertbackground=self.ACCENT,
                       relief='flat', bd=4)
        ent.pack(side='right', padx=(0,4))
        ent.bind('<Return>', lambda e, k=key: self._entry_commit(k))

        # Slider
        steps  = int(round((hi - lo) / res))
        slider = tk.Scale(frame, from_=lo, to=hi, resolution=res,
                          orient='horizontal', length=340,
                          bg=self.BG, fg=self.TEXT,
                          troughcolor=self.PANEL,
                          activebackground=self.ACCENT,
                          highlightthickness=0, bd=0,
                          showvalue=False,
                          command=lambda v, k=key: self._on_slide(k, v))
        slider.pack(side='left', padx=(8,4), fill='x', expand=True)
        slider.set(0.0)
        self.sliders[key] = slider

    # ── Logic ──────────────────────────────────
    def _on_slide(self, key, value):
        fv = float(value)
        fmt = f'{fv:.3f}' if 'm' in key else f'{fv:.1f}'
        self.value_vars[key].set(fmt)
        self.entry_vars[key].set(fmt)
        self._solve_and_publish()

    def _entry_commit(self, key):
        try:
            lo, hi, _ = self.RANGES[key]
            v = float(self.entry_vars[key].get())
            v = max(lo, min(hi, v))
            self.sliders[key].set(v)
        except ValueError:
            pass

    def _get_pose(self):
        x     = self.sliders['X (m)'].get()
        y     = self.sliders['Y (m)'].get()
        z     = self.sliders['Z (m)'].get()
        roll  = np.radians(self.sliders['Roll (°)'].get())
        pitch = np.radians(self.sliders['Pitch (°)'].get())
        yaw   = np.radians(self.sliders['Yaw (°)'].get())
        return np.array([x, y, z]), roll, pitch, yaw

    def _solve_and_publish(self):
        pos, roll, pitch, yaw = self._get_pose()
        R_tar = rpy_to_matrix(roll, pitch, yaw)

        base_y = self.node.current_slider
        q, reachable = inverse_kinematics(pos, R_tar, base_y)

        if not reachable:
            self.ik_var.set('⚠  Target out of reach')
            for i, v in enumerate(self.joint_vars):
                v.set('—')
            return

        self.ik_var.set('')
        for i, v in enumerate(self.joint_vars):
            v.set(f'{np.degrees(q[i]):.1f}°')

        self.last_q = q.tolist()
        self.node.publish(q, base_y)

    def _publish_now(self):
        self._solve_and_publish()

    def _sync_from_robot(self):
        """Read current joint state → FK → set sliders to current EE pose."""
        if not self.node.received:
            messagebox.showwarning('No Data', 'No /joint_states received yet.')
            return
        q      = self.node.current_joints
        base_y = self.node.current_slider
        T      = forward_kinematics(q, base_y)
        pos    = T[:3, 3]
        roll, pitch, yaw = matrix_to_rpy(T[:3, :3])

        self.sliders['X (m)'].set(round(float(pos[0]), 3))
        self.sliders['Y (m)'].set(round(float(pos[1]), 3))
        self.sliders['Z (m)'].set(round(float(pos[2]), 3))
        self.sliders['Roll (°)'].set(round(float(np.degrees(roll)),  1))
        self.sliders['Pitch (°)'].set(round(float(np.degrees(pitch)), 1))
        self.sliders['Yaw (°)'].set(round(float(np.degrees(yaw)),   1))

    def _wait_for_joints(self):
        """Poll until joint_states arrives, then update status dot."""
        if self.node.received:
            self.status_dot.config(fg=self.GREEN)
            self.status_lbl.config(text='/joint_states ✓', fg=self.GREEN)
            self._sync_from_robot()
        else:
            self.root.after(200, self._wait_for_joints)

    def run(self):
        self.root.mainloop()

# ──────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────
def main():
    rclpy.init()
    ros_node = GUINode()

    # Spin ROS in background thread
    spin_thread = threading.Thread(
        target=rclpy.spin, args=(ros_node,), daemon=True)
    spin_thread.start()

    app = App(ros_node)
    app.run()

    ros_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()