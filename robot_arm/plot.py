#!/usr/bin/env python3
"""
EE Pose → /goto_position GUI
─────────────────────────────
Sliders: X / Y / Z / Roll / Pitch / Yaw + Speed
SEND publishes JSON String to /goto_position.
arm_controller.py (plan node) handles motion execution.
Sliders update the IK preview only — robot does NOT move until SEND is pressed.
"""

import json
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String

# ── Robot constants (must match arm_controller.py) ─────────
L1, L2, L3 = 0.28787, 0.26096, 0.26136
D6          = 0.07074
J4_OFFSET_Y = 0.02175
A1          = 0.020885

Q_MIN = np.radians([-180.0, -180.0, -120.0, -180.0, -100.0, -360.0])
Q_MAX = np.radians([ 180.0,   30.0,  150.0,  180.0,  100.0,  360.0])

# ── Kinematics helpers ─────────────────────────────────────
def get_transform(theta, d, a, alpha):
    c, s   = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ c, -s*ca,  s*sa, a*c],
        [ s,  c*ca, -c*sa, a*s],
        [ 0,    sa,    ca,   d],
        [ 0,     0,     0,   1],
    ])

def forward_kinematics(q, base_y=0.0):
    """Gamma-corrected FK — matches arm_controller.forward_kinematics_matrices."""
    q1, q2, q3, q4, q5, q6 = q
    gamma  = np.arctan2(J4_OFFSET_Y, L3)
    l3_eff = np.sqrt(L3**2 + J4_OFFSET_Y**2)
    dh = [
        [q1,                   L1,     A1,  -np.pi/2],
        [q2,                    0,     L2,   0       ],
        [q3 + np.pi/2 - gamma,  0,      0,   np.pi/2 ],
        [q4,               l3_eff,      0,  -np.pi/2 ],
        [q5 + gamma,            0,      0,   np.pi/2 ],
        [q6,                   D6,      0,   0        ],
    ]
    T = np.eye(4)
    T[1, 3] = base_y
    for p in dh:
        T = T @ get_transform(*p)
    return T

def inverse_kinematics(target_pos, target_orient, base_y=0.0):
    """Returns (q[6] in radians, reachable: bool)."""
    local_pos       = target_pos - np.array([0.0, base_y, 0.0])
    xc, yc, zc     = local_pos - D6 * target_orient[:, 2]
    q1              = np.arctan2(yc, xc)
    r               = np.sqrt(xc**2 + yc**2)
    s               = zc - L1
    D_sq            = r**2 + s**2
    cos_q3          = (D_sq - L2**2 - L3**2) / (2 * L2 * L3)
    reachable       = True
    if cos_q3 > 1.0 or cos_q3 < -1.0:
        reachable = False
        cos_q3 = np.clip(cos_q3, -1.0, 1.0)
    sin_q3 = np.sqrt(max(0.0, 1.0 - cos_q3**2))
    q3     = np.arctan2(sin_q3, cos_q3)
    beta   = np.arctan2(L3 * np.sin(q3), L2 + L3 * np.cos(q3))
    q2     = np.arctan2(-s, r) - beta

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
    if np.abs(R36[2, 2]) > 0.9999:
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

# ── ROS2 Node ──────────────────────────────────────────────
class GUINode(Node):
    def __init__(self):
        super().__init__('ee_pose_gui')
        self.goto_pub = self.create_publisher(String, '/goto_position', 10)
        self.sub      = self.create_subscription(
            JointState, '/motor', self._joint_cb, 10)
        self.current_joints = [0.0] * 6
        self.current_slider = 0.0
        self.received       = False

    def _joint_cb(self, msg):
        if len(msg.position) >= 7:
            self.current_slider = msg.position[0]
            self.current_joints = list(msg.position[1:7])
            self.received = True

    def send_goto(self, joints_deg, rail_mm, speed_pct, control_mode):
        """Publish JSON task list to /goto_position."""
        payload = [{
            'label':        'move_point',
            'sequence':     1,
            'joint_1':      float(joints_deg[0]),
            'joint_2':      float(joints_deg[1]),
            'joint_3':      float(joints_deg[2]),
            'joint_4':      float(joints_deg[3]),
            'joint_5':      float(joints_deg[4]),
            'joint_6':      float(joints_deg[5]),
            'rail':         float(rail_mm),   # arm_controller divides by 1000 → metres
            'speed':        int(speed_pct),
            'delay':        None,
            'gripper':      0,
            'control_mode': control_mode,
        }]
        msg      = String()
        msg.data = json.dumps(payload)
        self.goto_pub.publish(msg)

# ── GUI ────────────────────────────────────────────────────
class App:
    POSE_RANGES = {
        'X (m)':     (-0.8,  0.8, 0.001),
        'Y (m)':     (-0.8,  0.8, 0.001),
        'Z (m)':     ( 0.0,  1.2, 0.001),
        'Roll (°)':  (-180, 180,  0.1  ),
        'Pitch (°)': (-180, 180,  0.1  ),
        'Yaw (°)':   (-180, 180,  0.1  ),
    }
    KEYS = list(POSE_RANGES.keys())

    BG     = '#0f1117'
    PANEL  = '#1a1d27'
    ACCENT = '#00e5ff'
    PINK   = '#ff4081'
    TEXT   = '#e8eaf6'
    MUTED  = '#555870'
    GREEN  = '#69ff6e'
    RED    = '#ff5252'

    def __init__(self, ros_node: GUINode):
        self.node   = ros_node
        self.last_q = list(ros_node.current_joints)
        self._ik_ok = False

        self.root = tk.Tk()
        self.root.title('EE Pose → /goto_position')
        self.root.configure(bg=self.BG)
        self.root.resizable(False, False)
        self._build_ui()
        self._wait_for_joints()

    # ── UI construction ──────────────────────────────────────
    def _build_ui(self):
        root = self.root

        # Header
        hdr = tk.Frame(root, bg=self.BG)
        hdr.pack(fill='x', padx=20, pady=(18, 4))
        tk.Label(hdr, text='EE POSE → /goto_position',
                 font=('Courier New', 13, 'bold'),
                 fg=self.ACCENT, bg=self.BG).pack(side='left')
        self.status_dot = tk.Label(hdr, text='●', font=('Courier New', 14),
                                   fg=self.RED, bg=self.BG)
        self.status_dot.pack(side='right')
        self.status_lbl = tk.Label(hdr, text='waiting for /motor…',
                                   font=('Courier New', 9),
                                   fg=self.MUTED, bg=self.BG)
        self.status_lbl.pack(side='right', padx=(0, 6))
        tk.Frame(root, bg=self.ACCENT, height=1).pack(fill='x', padx=20, pady=(0, 10))

        # Pose sliders — tabbed
        style = ttk.Style()
        style.theme_use('default')
        style.configure('D.TNotebook', background=self.BG, borderwidth=0)
        style.configure('D.TNotebook.Tab',
                        background=self.PANEL, foreground=self.MUTED,
                        font=('Courier New', 10, 'bold'), padding=[16, 6])
        style.map('D.TNotebook.Tab',
                  background=[('selected', self.BG)],
                  foreground=[('selected', self.ACCENT)])
        nb = ttk.Notebook(root, style='D.TNotebook')
        nb.pack(padx=20, fill='both')

        self.sliders    = {}
        self.value_vars = {}
        self.entry_vars = {}

        pos_tab = tk.Frame(nb, bg=self.BG)
        nb.add(pos_tab, text='  POSITION  ')
        for key in self.KEYS[:3]:
            self._add_slider_row(pos_tab, key)

        rot_tab = tk.Frame(nb, bg=self.BG)
        nb.add(rot_tab, text='  ORIENTATION  ')
        for key in self.KEYS[3:]:
            self._add_slider_row(rot_tab, key)

        # Speed + mode panel
        ctrl = tk.Frame(root, bg=self.PANEL)
        ctrl.pack(padx=20, pady=(10, 0), fill='x')

        # Speed row
        spd_row = tk.Frame(ctrl, bg=self.PANEL)
        spd_row.pack(fill='x', padx=8, pady=(6, 2))
        tk.Label(spd_row, text='SPEED',
                 font=('Courier New', 9, 'bold'),
                 fg=self.MUTED, bg=self.PANEL, width=8, anchor='w').pack(side='left')
        self.speed_var = tk.IntVar(value=50)
        tk.Scale(spd_row, from_=5, to=100, resolution=5,
                 orient='horizontal', length=300,
                 variable=self.speed_var,
                 bg=self.PANEL, fg=self.TEXT,
                 troughcolor=self.BG, activebackground=self.PINK,
                 highlightthickness=0, bd=0, showvalue=False).pack(
                     side='left', padx=(6, 4), fill='x', expand=True)
        tk.Label(spd_row, textvariable=self.speed_var,
                 width=4, font=('Courier New', 10, 'bold'),
                 fg=self.PINK, bg=self.PANEL).pack(side='left')
        tk.Label(spd_row, text='%',
                 font=('Courier New', 9), fg=self.MUTED, bg=self.PANEL).pack(side='left')

        # Mode row
        mode_row = tk.Frame(ctrl, bg=self.PANEL)
        mode_row.pack(fill='x', padx=8, pady=(0, 6))
        tk.Label(mode_row, text='MODE',
                 font=('Courier New', 9, 'bold'),
                 fg=self.MUTED, bg=self.PANEL, width=8, anchor='w').pack(side='left')
        self.mode_var = tk.StringVar(value='effector')
        for val, lbl in [('effector', 'Effector  (Cartesian path)'),
                         ('joint',    'Joint  (direct joint space)')]:
            tk.Radiobutton(mode_row, text=lbl, variable=self.mode_var, value=val,
                           font=('Courier New', 9),
                           fg=self.TEXT, bg=self.PANEL,
                           selectcolor=self.BG,
                           activebackground=self.PANEL,
                           activeforeground=self.ACCENT).pack(side='left', padx=10)

        # Joint readout
        jnt = tk.Frame(root, bg=self.PANEL)
        jnt.pack(padx=20, pady=(10, 0), fill='x')
        tk.Label(jnt, text=' JOINT SOLUTION  (IK preview — slider change only)',
                 font=('Courier New', 9, 'bold'),
                 fg=self.MUTED, bg=self.PANEL).grid(
                     row=0, column=0, columnspan=6, sticky='w', padx=8, pady=(6, 2))
        self.joint_vars = []
        for i in range(6):
            v = tk.StringVar(value='—')
            self.joint_vars.append(v)
            col = tk.Frame(jnt, bg=self.PANEL)
            col.grid(row=1, column=i, padx=8, pady=(0, 4))
            tk.Label(col, text=f'J{i+1}', font=('Courier New', 8),
                     fg=self.MUTED, bg=self.PANEL).pack()
            tk.Label(col, textvariable=v, font=('Courier New', 10, 'bold'),
                     fg=self.PINK, bg=self.PANEL, width=7).pack()
        self.ik_var = tk.StringVar(value='')
        tk.Label(jnt, textvariable=self.ik_var,
                 font=('Courier New', 9), fg=self.RED, bg=self.PANEL).grid(
                     row=2, column=0, columnspan=6, sticky='w', padx=8, pady=(0, 6))

        # Buttons
        btn = tk.Frame(root, bg=self.BG)
        btn.pack(padx=20, pady=14, fill='x')
        tk.Button(btn, text='SYNC FROM ROBOT',
                  font=('Courier New', 10, 'bold'),
                  bg=self.PANEL, fg=self.ACCENT,
                  activebackground=self.ACCENT, activeforeground=self.BG,
                  bd=0, padx=16, pady=8, cursor='hand2',
                  command=self._sync_from_robot).pack(side='left', padx=(0, 10))
        self.send_btn = tk.Button(btn, text='SEND TO ROBOT',
                  font=('Courier New', 10, 'bold'),
                  bg=self.ACCENT, fg=self.BG,
                  activebackground='#00bcd4', activeforeground=self.BG,
                  bd=0, padx=24, pady=8, cursor='hand2',
                  command=self._send_to_robot)
        self.send_btn.pack(side='left')

        # Footer
        tk.Frame(root, bg=self.MUTED, height=1).pack(fill='x', padx=20, pady=(0, 4))
        tk.Label(root,
                 text='Sliders preview IK only.  SEND publishes JSON to /goto_position.',
                 font=('Courier New', 8), fg=self.MUTED, bg=self.BG).pack(pady=(0, 10))

    def _add_slider_row(self, parent, key):
        lo, hi, res = self.POSE_RANGES[key]
        row = tk.Frame(parent, bg=self.BG)
        row.pack(fill='x', padx=16, pady=6)

        tk.Label(row, text=key, width=11, anchor='w',
                 font=('Courier New', 10, 'bold'),
                 fg=self.TEXT, bg=self.BG).pack(side='left')

        val_var = tk.StringVar(value='0.000')
        self.value_vars[key] = val_var
        tk.Label(row, textvariable=val_var, width=8, anchor='e',
                 font=('Courier New', 10),
                 fg=self.ACCENT, bg=self.BG).pack(side='right', padx=(6, 0))

        ent_var = tk.StringVar()
        self.entry_vars[key] = ent_var
        ent = tk.Entry(row, textvariable=ent_var, width=7,
                       font=('Courier New', 9),
                       bg=self.PANEL, fg=self.TEXT,
                       insertbackground=self.ACCENT, relief='flat', bd=4)
        ent.pack(side='right', padx=(0, 4))
        ent.bind('<Return>', lambda e, k=key: self._entry_commit(k))

        slider = tk.Scale(row, from_=lo, to=hi, resolution=res,
                          orient='horizontal', length=340,
                          bg=self.BG, fg=self.TEXT,
                          troughcolor=self.PANEL, activebackground=self.ACCENT,
                          highlightthickness=0, bd=0, showvalue=False,
                          command=lambda v, k=key: self._on_slide(k, v))
        slider.pack(side='left', padx=(8, 4), fill='x', expand=True)
        slider.set(0.0)
        self.sliders[key] = slider

    # ── Logic ───────────────────────────────────────────────
    def _on_slide(self, key, value):
        fv  = float(value)
        fmt = f'{fv:.3f}' if 'm' in key else f'{fv:.1f}'
        self.value_vars[key].set(fmt)
        self.entry_vars[key].set(fmt)
        self._update_ik_display()   # preview only, no publish

    def _entry_commit(self, key):
        try:
            lo, hi, _ = self.POSE_RANGES[key]
            v = float(self.entry_vars[key].get())
            self.sliders[key].set(max(lo, min(hi, v)))
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

    def _update_ik_display(self):
        """Solve IK and update joint readout — does NOT publish."""
        pos, roll, pitch, yaw = self._get_pose()
        q, reachable = inverse_kinematics(pos, rpy_to_matrix(roll, pitch, yaw),
                                          self.node.current_slider)
        if not reachable:
            self.ik_var.set('⚠  Target out of reach')
            self.send_btn.config(state='disabled', bg=self.MUTED)
            for v in self.joint_vars:
                v.set('—')
            self._ik_ok = False
            return

        self.ik_var.set('')
        self.send_btn.config(state='normal', bg=self.ACCENT)
        for i, v in enumerate(self.joint_vars):
            v.set(f'{np.degrees(q[i]):.1f}°')
        self.last_q = q.tolist()
        self._ik_ok = True

    def _send_to_robot(self):
        """Solve IK and publish to /goto_position."""
        pos, roll, pitch, yaw = self._get_pose()
        q, reachable = inverse_kinematics(pos, rpy_to_matrix(roll, pitch, yaw),
                                          self.node.current_slider)
        if not reachable:
            messagebox.showwarning('IK Failed', 'Target is out of reach — cannot send.')
            return

        joints_deg = np.degrees(q).tolist()
        rail_mm    = self.node.current_slider * 1000.0  # current rail position in mm
        speed_pct  = self.speed_var.get()
        mode       = self.mode_var.get()

        self.node.send_goto(joints_deg, rail_mm, speed_pct, mode)
        self.node.get_logger().info(
            f'[plot] sent xyz={np.round(pos,3)} '
            f'rpy={np.round(np.degrees([roll,pitch,yaw]),1)} '
            f'speed={speed_pct}% mode={mode}'
        )

    def _sync_from_robot(self):
        """FK of current joints → set sliders to current EE pose."""
        if not self.node.received:
            messagebox.showwarning('No Data', 'No /motor data received yet.')
            return
        T             = forward_kinematics(self.node.current_joints, self.node.current_slider)
        pos           = T[:3, 3]
        roll, pitch, yaw = matrix_to_rpy(T[:3, :3])
        self.sliders['X (m)'].set(round(float(pos[0]),            3))
        self.sliders['Y (m)'].set(round(float(pos[1]),            3))
        self.sliders['Z (m)'].set(round(float(pos[2]),            3))
        self.sliders['Roll (°)'].set( round(float(np.degrees(roll)),  1))
        self.sliders['Pitch (°)'].set(round(float(np.degrees(pitch)), 1))
        self.sliders['Yaw (°)'].set(  round(float(np.degrees(yaw)),   1))

    def _wait_for_joints(self):
        if self.node.received:
            self.status_dot.config(fg=self.GREEN)
            self.status_lbl.config(text='/motor ✓', fg=self.GREEN)
            self._sync_from_robot()
        else:
            self.root.after(200, self._wait_for_joints)

    def run(self):
        self.root.mainloop()

# ── Entry point ────────────────────────────────────────────
def main():
    rclpy.init()
    ros_node = GUINode()

    threading.Thread(target=rclpy.spin, args=(ros_node,), daemon=True).start()

    app = App(ros_node)
    app.run()

    ros_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
