#!/usr/bin/env python3
"""
Robot Arm Jog UI
Publishes jog commands to /goto_position topic in both
Joint Space (instant_jog_joint) and Task Space (instant_jog_task).
"""

import sys
import json
import threading
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import JointState

try:
    import tkinter as tk
    from tkinter import ttk, font
except ImportError:
    print("tkinter not found. Install it with: sudo apt install python3-tk")
    sys.exit(1)


# ─────────────────────────────────────────────
#  ROS2 Publisher Node
# ─────────────────────────────────────────────
class JogPublisher(Node):
    def __init__(self):
        super().__init__('arm_jog_ui_node')
        self.publisher = self.create_publisher(String, '/goto_position', 10)
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self._joint_cb, 10)

        self.current_joints = [0.0] * 6
        self.current_slider = 0.0
        self.has_data = False

    def _joint_cb(self, msg: JointState):
        if len(msg.position) >= 7:
            self.current_slider = msg.position[0]
            self.current_joints = list(msg.position[1:7])
            self.has_data = True

    def send_jog_joint(self, axis: str, direction: int, speed: int):
        payload = json.dumps([{
            "label": "jog",
            "control_mode": "joint",
            "axis": axis,
            "direction": direction,
            "speed": speed,
        }])
        msg = String()
        msg.data = payload
        self.publisher.publish(msg)
        self.get_logger().info(f"[JOG JOINT] axis={axis} dir={direction} speed={speed}%")

    def send_jog_task(self, axis: str, direction: int, speed: int,
                      tcp_x=0.0, tcp_y=0.0, tcp_z=0.0):
        payload = json.dumps([{
            "label": "jog",
            "control_mode": "effector",
            "axis": axis,
            "direction": direction,
            "speed": speed,
            "tcp_x": tcp_x,
            "tcp_y": tcp_y,
            "tcp_z": tcp_z,
        }])
        msg = String()
        msg.data = payload
        self.publisher.publish(msg)
        self.get_logger().info(f"[JOG TASK] axis={axis} dir={direction} speed={speed}%")


# ─────────────────────────────────────────────
#  Tkinter UI
# ─────────────────────────────────────────────
class JogUI:
    # Colour palette
    BG          = "#1e1e2e"
    PANEL_BG    = "#2a2a3e"
    ACCENT_BLUE = "#4fc3f7"
    ACCENT_GRN  = "#69f0ae"
    ACCENT_ORG  = "#ffb74d"
    BTN_POS     = "#1565c0"
    BTN_NEG     = "#b71c1c"
    BTN_HOV_POS = "#1e88e5"
    BTN_HOV_NEG = "#e53935"
    TEXT        = "#e0e0e0"
    TEXT_DIM    = "#9e9e9e"
    BORDER      = "#3a3a5e"

    def __init__(self, ros_node: JogPublisher):
        self.node = ros_node

        self.root = tk.Tk()
        self.root.title("Robot Arm Jog Controller")
        self.root.configure(bg=self.BG)
        self.root.resizable(False, False)

        self.speed_var = tk.IntVar(value=30)
        self.mode_var  = tk.StringVar(value="joint")   # "joint" | "task"
        self.status_var = tk.StringVar(value="Ready")

        # TCP offset vars (mm)
        self.tcp_x_var = tk.DoubleVar(value=0.0)
        self.tcp_y_var = tk.DoubleVar(value=0.0)
        self.tcp_z_var = tk.DoubleVar(value=0.0)

        self._build_ui()
        self._start_joint_display_update()

    # ── Build UI ──────────────────────────────
    def _build_ui(self):
        root = self.root
        pad = dict(padx=10, pady=6)

        # Title bar
        title_frame = tk.Frame(root, bg=self.BG)
        title_frame.pack(fill="x", padx=12, pady=(14, 4))
        tk.Label(title_frame, text="🤖  Robot Arm Jog Controller",
                 bg=self.BG, fg=self.ACCENT_BLUE,
                 font=("Helvetica", 17, "bold")).pack(side="left")
        self.status_lbl = tk.Label(title_frame, textvariable=self.status_var,
                                   bg=self.BG, fg=self.ACCENT_GRN,
                                   font=("Helvetica", 10))
        self.status_lbl.pack(side="right", padx=6)

        # ── Mode toggle button ───────────────
        mode_frame = tk.Frame(root, bg=self.BG)
        mode_frame.pack(fill="x", padx=12, pady=(0, 6))

        self.mode_btn = tk.Button(
            mode_frame,
            text="🦾  JOINT SPACE",
            bg=self.ACCENT_BLUE, fg=self.BG,
            activebackground=self.BTN_HOV_POS, activeforeground=self.BG,
            font=("Helvetica", 14, "bold"),
            relief="flat", bd=0,
            padx=20, pady=10,
            cursor="hand2",
            command=self._toggle_mode
        )
        self.mode_btn.pack(fill="x")

        self.mode_indicator = tk.Label(
            mode_frame, text="● Joint Space active",
            bg=self.BG, fg=self.ACCENT_BLUE,
            font=("Helvetica", 9)
        )
        self.mode_indicator.pack(anchor="center", pady=(2, 0))

        # ── Speed slider ─────────────────────
        spd_frame = self._panel(root, "Speed")
        spd_frame.pack(fill="x", padx=12, pady=(0, 4))
        tk.Label(spd_frame, text="Speed:", bg=self.PANEL_BG,
                 fg=self.TEXT, font=("Helvetica", 11)).pack(side="left", padx=8)
        self.speed_lbl = tk.Label(spd_frame, text="30%",
                                  bg=self.PANEL_BG, fg=self.ACCENT_GRN,
                                  font=("Helvetica", 12, "bold"), width=5)
        self.speed_lbl.pack(side="right", padx=10)
        slider = tk.Scale(spd_frame, from_=1, to=100, orient="horizontal",
                          variable=self.speed_var, bg=self.PANEL_BG,
                          fg=self.TEXT, highlightthickness=0,
                          troughcolor=self.BORDER, activebackground=self.ACCENT_BLUE,
                          length=320, command=self._on_speed_change)
        slider.pack(side="left", fill="x", expand=True, padx=4)

        # ── Main jog area (joint / task panels) ──
        self.jog_container = tk.Frame(root, bg=self.BG)
        self.jog_container.pack(fill="both", expand=True, padx=12, pady=4)

        self._build_joint_panel()
        self._build_task_panel()
        self._build_tcp_panel()       # TCP offset (task mode only)

        self._on_mode_change()        # show correct panel

        # ── Joint state readout ──────────────
        self._build_joint_readout(root)

    # ── Joint jog panel ──────────────────────
    def _build_joint_panel(self):
        self.joint_panel = self._panel(self.jog_container, "Joint Space Jog")

        joints = [
            ("Joint 1", "joint_1"), ("Joint 2", "joint_2"),
            ("Joint 3", "joint_3"), ("Joint 4", "joint_4"),
            ("Joint 5", "joint_5"), ("Joint 6", "joint_6"),
            ("Rail / Slider", "rail"),
        ]
        for i, (label, axis) in enumerate(joints):
            row = tk.Frame(self.joint_panel, bg=self.PANEL_BG)
            row.grid(row=i, column=0, sticky="ew", padx=8, pady=3)
            self.joint_panel.columnconfigure(0, weight=1)

            tk.Label(row, text=label, bg=self.PANEL_BG, fg=self.TEXT,
                     font=("Helvetica", 11), width=14, anchor="w").pack(side="left")

            self._jog_btn(row, "◀  –", axis, -1, "joint")
            self._jog_btn(row, "+  ▶", axis, +1, "joint")

    # ── Task jog panel ───────────────────────
    def _build_task_panel(self):
        self.task_panel = self._panel(self.jog_container, "Task Space Jog (End-Effector)")

        axes = [
            ("X  (Forward/Back)",   "x"),
            ("Y  (Left/Right)",     "y"),
            ("Z  (Up/Down)",        "z"),
            ("Roll  (Rx)",          "roll"),
            ("Pitch (Ry)",          "pitch"),
            ("Yaw   (Rz)",          "yaw"),
        ]
        colors = {
            "x": "#ef9a9a", "y": "#a5d6a7", "z": "#90caf9",
            "roll": "#ffe082", "pitch": "#ce93d8", "yaw": "#80cbc4",
        }
        for i, (label, axis) in enumerate(axes):
            row = tk.Frame(self.task_panel, bg=self.PANEL_BG)
            row.grid(row=i, column=0, sticky="ew", padx=8, pady=3)
            self.task_panel.columnconfigure(0, weight=1)

            c = colors.get(axis, self.TEXT)
            tk.Label(row, text=label, bg=self.PANEL_BG, fg=c,
                     font=("Helvetica", 11), width=20, anchor="w").pack(side="left")

            self._jog_btn(row, "◀  –", axis, -1, "task")
            self._jog_btn(row, "+  ▶", axis, +1, "task")

    # ── TCP offset panel ─────────────────────
    def _build_tcp_panel(self):
        self.tcp_panel = self._panel(self.jog_container, "TCP Offset  (mm)")

        for label, var in [("TCP X:", self.tcp_x_var),
                           ("TCP Y:", self.tcp_y_var),
                           ("TCP Z:", self.tcp_z_var)]:
            row = tk.Frame(self.tcp_panel, bg=self.PANEL_BG)
            row.pack(fill="x", padx=8, pady=2)
            tk.Label(row, text=label, bg=self.PANEL_BG, fg=self.TEXT_DIM,
                     font=("Helvetica", 10), width=8, anchor="w").pack(side="left")
            spinbox = tk.Spinbox(row, from_=-500, to=500, increment=1,
                                 textvariable=var, width=8,
                                 bg=self.BORDER, fg=self.TEXT,
                                 buttonbackground=self.PANEL_BG,
                                 highlightthickness=0,
                                 font=("Helvetica", 10))
            spinbox.pack(side="left", padx=4)

    # ── Joint state readout ──────────────────
    def _build_joint_readout(self, parent):
        frame = self._panel(parent, "Current Joint Positions  (degrees / m)")
        frame.pack(fill="x", padx=12, pady=(4, 12))

        self.joint_labels = []
        names = ["J1", "J2", "J3", "J4", "J5", "J6", "Rail"]
        inner = tk.Frame(frame, bg=self.PANEL_BG)
        inner.pack(padx=6, pady=4)
        for col, name in enumerate(names):
            tk.Label(inner, text=name, bg=self.PANEL_BG, fg=self.TEXT_DIM,
                     font=("Helvetica", 9), width=8).grid(row=0, column=col, padx=3)
            lbl = tk.Label(inner, text="—", bg=self.PANEL_BG,
                           fg=self.ACCENT_GRN,
                           font=("Courier", 10, "bold"), width=8)
            lbl.grid(row=1, column=col, padx=3, pady=2)
            self.joint_labels.append(lbl)

    # ── Helpers ──────────────────────────────
    def _panel(self, parent, title: str) -> tk.Frame:
        outer = tk.Frame(parent, bg=self.BG)
        tk.Label(outer, text=title, bg=self.BG, fg=self.TEXT_DIM,
                 font=("Helvetica", 9)).pack(anchor="w", padx=4)
        inner = tk.Frame(outer, bg=self.PANEL_BG,
                         highlightbackground=self.BORDER,
                         highlightthickness=1)
        inner.pack(fill="both", expand=True, padx=2, pady=(0, 2))
        return inner

    def _jog_btn(self, parent, text, axis, direction, mode):
        color = self.BTN_NEG if direction < 0 else self.BTN_POS
        hover  = self.BTN_HOV_NEG if direction < 0 else self.BTN_HOV_POS

        btn = tk.Button(
            parent, text=text,
            bg=color, fg="white",
            activebackground=hover, activeforeground="white",
            font=("Helvetica", 10, "bold"),
            relief="flat", bd=0,
            padx=14, pady=5,
            cursor="hand2",
            command=lambda a=axis, d=direction, m=mode: self._on_jog(a, d, m)
        )
        btn.pack(side="left", padx=4)
        # Hover effect
        btn.bind("<Enter>", lambda e, b=btn, h=hover: b.config(bg=h))
        btn.bind("<Leave>", lambda e, b=btn, c=color: b.config(bg=c))

    def _toggle_mode(self):
        self.mode_var.set("task" if self.mode_var.get() == "joint" else "joint")
        self._on_mode_change()

    def _on_mode_change(self):
        mode = self.mode_var.get()
        self.joint_panel.master.pack_forget()
        self.task_panel.master.pack_forget()
        self.tcp_panel.master.pack_forget()

        if mode == "joint":
            self.joint_panel.master.pack(fill="both", expand=True)
            self.mode_btn.config(
                text="🦾  JOINT SPACE  —  click to switch to Task Space →",
                bg=self.ACCENT_BLUE, fg=self.BG,
                activebackground=self.BTN_HOV_POS)
            self.mode_indicator.config(
                text="● Joint Space active", fg=self.ACCENT_BLUE)
        else:
            self.task_panel.master.pack(fill="both", expand=True)
            self.tcp_panel.master.pack(fill="x", pady=(4, 0))
            self.mode_btn.config(
                text="🖐  TASK SPACE  —  click to switch to Joint Space →",
                bg=self.ACCENT_ORG, fg=self.BG,
                activebackground="#ffa726")
            self.mode_indicator.config(
                text="● Task Space (End-Effector) active", fg=self.ACCENT_ORG)

    def _on_speed_change(self, _=None):
        self.speed_lbl.config(text=f"{self.speed_var.get()}%")

    def _on_jog(self, axis: str, direction: int, mode: str):
        speed = self.speed_var.get()
        if mode == "joint":
            self.node.send_jog_joint(axis, direction, speed)
            self.status_var.set(
                f"▶ Joint jog  {axis}  {'↑+' if direction>0 else '↓–'}  {speed}%")
        else:
            tcp_x = self.tcp_x_var.get()
            tcp_y = self.tcp_y_var.get()
            tcp_z = self.tcp_z_var.get()
            self.node.send_jog_task(axis, direction, speed, tcp_x, tcp_y, tcp_z)
            self.status_var.set(
                f"▶ Task jog  {axis}  {'↑+' if direction>0 else '↓–'}  {speed}%")

    def _start_joint_display_update(self):
        self._update_joint_display()

    def _update_joint_display(self):
        import math
        if self.node.has_data:
            for i, val in enumerate(self.node.current_joints):
                self.joint_labels[i].config(text=f"{math.degrees(val):+.1f}°")
            self.joint_labels[6].config(
                text=f"{self.node.current_slider:.3f}m")
        self.root.after(200, self._update_joint_display)

    def run(self):
        self.root.mainloop()


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────
def main():
    rclpy.init()
    ros_node = JogPublisher()

    # Spin ROS in background thread
    ros_thread = threading.Thread(
        target=rclpy.spin, args=(ros_node,), daemon=True)
    ros_thread.start()

    print("✅  ROS2 Jog UI Node started")
    print("📡  Publishing to: /goto_position")
    print("📥  Subscribing to: /joint_states")

    ui = JogUI(ros_node)
    ui.run()

    ros_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()