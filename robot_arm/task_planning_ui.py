
import tkinter as tk
from tkinter import ttk
import json
import time
import threading
import argparse
from collections import deque

# ROS2 is optional — imported at runtime only when --ros2 flag is passed
#ros2 topic pub --once /task_planner/initial_state std_msgs/String   '{"data": "[\"at(home)\", \"hand_empty\"]"}'
#ros2 topic pub --once /task_planner/goal_state std_msgs/msg/String   "{data: '[\"at(pick_point)\", \"cup_centered\",\"hand_empty\"]'}"


try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
    _ROS2_AVAILABLE = True
except ImportError:
    _ROS2_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────
# Planner engine (inline so this file is self-contained)
# ─────────────────────────────────────────────────────────────────

def _state_key(s): return "|".join(sorted(s))

def _apply_op(state: set, op: dict) -> set:
    return (state - set(op["delete_effects"])) | set(op["add_effects"])

def _is_goal(state: set, goal: list) -> bool:
    return all(p in state for p in goal)

def plan_bfs(initial: list, goal: list, operators: list, max_depth=30):
    start = set(initial)
    if _is_goal(start, goal):
        return [], [{"state": sorted(start), "action": "START", "step": 0}]
    queue = deque([{"state": start, "plan": [], "step": 0}])
    visited = {_state_key(start)}
    explored = [{"state": sorted(start), "action": "START", "step": 0}]
    while queue:
        node = queue.popleft()
        if len(node["plan"]) >= max_depth:
            continue
        for op in operators:
            if not all(p in node["state"] for p in op["preconditions"]):
                continue
            nxt = _apply_op(node["state"], op)
            key = _state_key(nxt)
            if key in visited:
                continue
            visited.add(key)
            child = {"state": nxt, "plan": node["plan"] + [op["name"]], "step": len(node["plan"]) + 1}
            explored.append({"state": sorted(nxt), "action": op["name"], "step": child["step"]})
            if _is_goal(nxt, goal):
                return child["plan"], explored
            queue.append(child)
    return None, explored

def trace_plan(initial: list, plan: list, operators: list):
    op_map = {o["name"]: o for o in operators}
    state = set(initial)
    trace = [{"step": 0, "action": "START", "state": sorted(state)}]
    for i, name in enumerate(plan, 1):
        state = _apply_op(state, op_map[name])
        trace.append({"step": i, "action": name, "state": sorted(state)})
    return trace

# ═════════════════════════════════════════════════════════════════
#  USER CONFIGURATION — edit this section to set up your problem
# ═════════════════════════════════════════════════════════════════

DEFAULT_SCENARIO = "Cup Pick and Place"

SCENARIOS = {

    # ── Cup Pick and Place ────────────────────────────────────────
    "Cup Pick and Place": {
        "initial_state": [
            "at(home)",
            "hand_empty",
        ],
        "goal_predicates": [
            "at(home)",
            "hand_empty",
            "cup_at(destination)",
        ],
        "operators": [
            {
                "name": "move_to_search",
                "preconditions":  ["at(home)", "hand_empty"],
                "add_effects":    ["at(search)", "see(cup)"],
                "delete_effects": ["at(home)"],
            },
            {
                "name": "search_cup",
                "preconditions":  ["at(search)", "see(cup)", "hand_empty"],
                "add_effects":    ["cup_centered", "cup_reachable"],
                "delete_effects": [],
            },
            {
                "name": "align_perpendicular",
                "preconditions":  ["at(search)", "see(cup)", "cup_reachable", "hand_empty"],
                "add_effects":    ["at(perpendicular)"],
                "delete_effects": ["at(search)"],
            },
            {
                "name": "move_to_pick",
                "preconditions":  ["at(perpendicular)", "see(cup)", "cup_reachable", "hand_empty"],
                "add_effects":    ["at(pick_point)"],
                "delete_effects": ["at(perpendicular)"],
            },
            {
                "name": "pickup_cup",
                "preconditions":  ["at(pick_point)", "hand_empty", "cup_centered"],
                "add_effects":    ["holding(cup)"],
                "delete_effects": ["hand_empty", "cup_centered"],
            },
            {
                "name": "move_to_destination",
                "preconditions":  ["at(pick_point)", "holding(cup)"],
                "add_effects":    ["at(destination)"],
                "delete_effects": ["at(pick_point)", "see(cup)", "cup_reachable"],
            },
            {
                "name": "putdown_cup",
                "preconditions":  ["at(destination)", "holding(cup)"],
                "add_effects":    ["hand_empty", "cup_at(destination)"],
                "delete_effects": ["holding(cup)"],
            },
            {
                "name": "move_home",
                "preconditions":  ["at(destination)", "hand_empty", "cup_at(destination)"],
                "add_effects":    ["at(home)"],
                "delete_effects": ["at(destination)"],
            },
        ],
    },

}

# ═════════════════════════════════════════════════════════════════
#  END OF USER CONFIGURATION
# ═════════════════════════════════════════════════════════════════

EXAMPLES = SCENARIOS

# ─────────────────────────────────────────────────────────────────
# Colours
# ─────────────────────────────────────────────────────────────────

C = {
    "bg":       "#ffffff",
    "bg2":      "#f5f4f0",
    "bg3":      "#eceae4",
    "border":   "#d3d1c7",
    "text":     "#1a1a18",
    "muted":    "#73726c",
    "blue_bg":  "#e6f1fb",
    "blue_fg":  "#0c447c",
    "blue_bd":  "#85b7eb",
    "green_bg": "#eaf3de",
    "green_fg": "#27500a",
    "green_bd": "#97c459",
    "red_bg":   "#fcebeb",
    "red_fg":   "#791f1f",
    "red_bd":   "#f09595",
    "amber_bg": "#faeeda",
    "amber_fg": "#633806",
}

FONT       = ("Segoe UI", 10)
FONT_BOLD  = ("Segoe UI", 10, "bold")
FONT_MONO  = ("Courier New", 10)
FONT_SMALL = ("Segoe UI", 9)
FONT_HEAD  = ("Segoe UI", 12, "bold")

# ─────────────────────────────────────────────────────────────────
# Reusable widget: PredicatePanel (display-only)
# ─────────────────────────────────────────────────────────────────

class PredicatePanel(tk.Frame):
    def __init__(self, master, label, color="blue", **kw):
        super().__init__(master, bg=C["bg"], **kw)
        self._items: list[str] = []

        acc_bg = C["blue_bg"]  if color == "blue"  else C["green_bg"]
        acc_fg = C["blue_fg"]  if color == "blue"  else C["green_fg"]
        acc_bd = C["blue_bd"]  if color == "blue"  else C["green_bd"]
        self._acc = (acc_bg, acc_fg, acc_bd)

        # ── header ──────────────────────────────────────────────
        hdr = tk.Frame(self, bg=C["bg"])
        hdr.pack(fill="x", pady=(0, 4))
        tk.Label(hdr, text=label.upper(), font=("Segoe UI", 8, "bold"),
                 fg=C["muted"], bg=C["bg"], anchor="w").pack(side="left")
        self._count_lbl = tk.Label(hdr, text="", font=FONT_SMALL,
                                   fg=C["muted"], bg=C["bg"])
        self._count_lbl.pack(side="right")

        # ── chips frame ──────────────────────────────────────────
        self._chips_outer = tk.Frame(self, bg=C["bg2"],
                                     highlightbackground=C["border"], highlightthickness=1)
        self._chips_outer.pack(fill="x")
        self._chips_frame = tk.Frame(self._chips_outer, bg=C["bg2"])
        self._chips_frame.pack(fill="x", padx=6, pady=6)

        self._rebuild_chips()

    # ── public API ───────────────────────────────────────────────

    def get(self) -> list[str]:
        return list(self._items)

    def set(self, items: list[str]):
        self._items = list(items)
        self._rebuild_chips()

    # ── internals ────────────────────────────────────────────────

    def _rebuild_chips(self):
        for w in self._chips_frame.winfo_children():
            w.destroy()

        if not self._items:
            tk.Label(self._chips_frame, text="No predicates",
                     font=FONT_SMALL, fg=C["muted"], bg=C["bg2"]).pack(anchor="w")
        else:
            row_frame = tk.Frame(self._chips_frame, bg=C["bg2"])
            row_frame.pack(fill="x")
            col = 0
            for p in self._items:
                chip = self._make_chip(row_frame, p)
                chip.grid(row=0, column=col, padx=(0, 4), pady=2, sticky="w")
                col += 1
                if col >= 4:
                    row_frame = tk.Frame(self._chips_frame, bg=C["bg2"])
                    row_frame.pack(fill="x")
                    col = 0

        self._count_lbl.config(text=f"{len(self._items)} predicate{'s' if len(self._items) != 1 else ''}")

    def _make_chip(self, parent, text: str):
        frame = tk.Frame(parent, bg=C["border"], padx=1, pady=1)
        inner = tk.Frame(frame, bg=C["bg"])
        inner.pack()
        tk.Label(inner, text=text, font=FONT_MONO, bg=C["bg"], fg=C["text"],
                 padx=6, pady=2).pack(side="left")
        return frame


# ─────────────────────────────────────────────────────────────────
# Main application window
# ─────────────────────────────────────────────────────────────────

class PlannerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("STRIPS Task Planner")
        self.geometry("900x700")
        self.minsize(800, 600)
        self.configure(bg=C["bg"])

        self._config = dict(EXAMPLES[DEFAULT_SCENARIO])
        self._config["operators"] = list(self._config["operators"])
        self._plan_result = None
        self._ros2_received = {}
        self._plan_execution = None   # active execution state
        self._ros2_bridge = None      # set by main() after bridge is created

        self._build_ui()
        self._load_config_to_ui()

    # ── UI construction ──────────────────────────────────────────

    def _build_ui(self):
        # Top toolbar (title only)
        toolbar = tk.Frame(self, bg=C["bg2"],
                           highlightbackground=C["border"], highlightthickness=1)
        toolbar.pack(fill="x")
        tk.Label(toolbar, text="STRIPS Task Planner", font=FONT_HEAD,
                 fg=C["text"], bg=C["bg2"]).pack(side="left", padx=12, pady=8)

        # ROS2 status bar (hidden until ROS2 is connected)
        self._ros2_bar = tk.Frame(self, bg=C["bg3"],
                                  highlightbackground=C["border"], highlightthickness=1)
        self._ros2_lbl = tk.Label(self._ros2_bar, text="", font=FONT_SMALL,
                                  fg=C["muted"], bg=C["bg3"])
        self._ros2_lbl.pack(side="left", padx=10, pady=4)
        self._ros2_dot = tk.Label(self._ros2_bar, text="●", font=FONT_SMALL,
                                  fg=C["muted"], bg=C["bg3"])
        self._ros2_dot.pack(side="right", padx=10, pady=4)

        # Notebook
        style = ttk.Style()
        style.configure("TNotebook", background=C["bg"])
        style.configure("TNotebook.Tab", font=FONT, padding=[12, 6])

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True)

        self._tab_config = tk.Frame(nb, bg=C["bg"])
        self._tab_result = tk.Frame(nb, bg=C["bg"])
        nb.add(self._tab_config, text="Configuration")
        nb.add(self._tab_result, text="Results")

        self._build_config_tab()
        self._build_result_tab()

    def _build_config_tab(self):
        canvas = tk.Canvas(self._tab_config, bg=C["bg"], highlightthickness=0)
        scroll = ttk.Scrollbar(self._tab_config, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        self._cfg_inner = tk.Frame(canvas, bg=C["bg"])
        win_id = canvas.create_window((0, 0), window=self._cfg_inner, anchor="nw")

        def on_resize(e):
            canvas.itemconfig(win_id, width=e.width)
        def on_frame_resize(e):
            canvas.configure(scrollregion=canvas.bbox("all"))

        canvas.bind("<Configure>", on_resize)
        self._cfg_inner.bind("<Configure>", on_frame_resize)
        canvas.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(-1*(e.delta//120), "units")))
        canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))

        pad = {"padx": 20, "pady": 6}

        # ── Initial state ──────────────────────────────────────
        self._initial_panel = PredicatePanel(self._cfg_inner, "Initial State", color="blue")
        self._initial_panel.pack(fill="x", **pad)

        ttk.Separator(self._cfg_inner, orient="horizontal").pack(fill="x", padx=20, pady=4)

        # ── Goal ───────────────────────────────────────────────
        self._goal_panel = PredicatePanel(self._cfg_inner, "Goal Predicates", color="green")
        self._goal_panel.pack(fill="x", **pad)

        ttk.Separator(self._cfg_inner, orient="horizontal").pack(fill="x", padx=20, pady=4)

        # ── Operators ──────────────────────────────────────────
        ops_hdr = tk.Frame(self._cfg_inner, bg=C["bg"])
        ops_hdr.pack(fill="x", padx=20, pady=(6, 2))
        tk.Label(ops_hdr, text="OPERATORS", font=("Segoe UI", 8, "bold"),
                 fg=C["muted"], bg=C["bg"]).pack(side="left")
        self._ops_count_lbl = tk.Label(ops_hdr, text="", font=FONT_SMALL,
                                       fg=C["muted"], bg=C["bg"])
        self._ops_count_lbl.pack(side="right")

        self._ops_frame = tk.Frame(self._cfg_inner, bg=C["bg"])
        self._ops_frame.pack(fill="x", padx=20, pady=(0, 6))

    def _build_result_tab(self):
        self._result_text = tk.Text(self._tab_result, font=FONT_MONO,
                                    bg=C["bg"], fg=C["text"], relief="flat",
                                    wrap="word", state="disabled", padx=16, pady=12)
        scroll = ttk.Scrollbar(self._tab_result, command=self._result_text.yview)
        self._result_text.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self._result_text.pack(fill="both", expand=True)

        self._result_text.tag_configure("head",  font=("Segoe UI", 11, "bold"), foreground=C["text"])
        self._result_text.tag_configure("step",  font=FONT_MONO, foreground=C["blue_fg"])
        self._result_text.tag_configure("state", font=FONT_MONO, foreground=C["muted"])
        self._result_text.tag_configure("goal",  font=FONT_MONO, foreground=C["green_fg"])
        self._result_text.tag_configure("ok",    font=FONT_BOLD, foreground=C["green_fg"])
        self._result_text.tag_configure("err",   font=FONT_BOLD, foreground=C["red_fg"])
        self._result_text.tag_configure("info",  font=FONT_SMALL, foreground=C["muted"])

        self._write_result("Waiting for planner results…\n")

    # ── Operators display ────────────────────────────────────────

    def _refresh_operators_ui(self):
        for w in self._ops_frame.winfo_children():
            w.destroy()
        ops = self._config.get("operators", [])
        self._ops_count_lbl.config(text=f"{len(ops)} operator{'s' if len(ops) != 1 else ''}")
        for i, op in enumerate(ops):
            self._build_operator_row(i, op)

    def _build_operator_row(self, idx: int, op: dict):
        row = tk.Frame(self._ops_frame, bg=C["bg2"],
                       highlightbackground=C["border"], highlightthickness=1)
        row.pack(fill="x", pady=(0, 4))

        hdr = tk.Frame(row, bg=C["bg2"])
        hdr.pack(fill="x", padx=8, pady=4)
        tk.Label(hdr, text=op["name"], font=FONT_MONO,
                 fg=C["text"], bg=C["bg2"]).pack(side="left")

        detail = tk.Frame(row, bg=C["bg2"])
        detail.pack(fill="x", padx=8, pady=(0, 6))
        self._chip_row(detail, "pre:",  op["preconditions"],  C["blue_bg"],  C["blue_fg"],  C["blue_bd"])
        self._chip_row(detail, "+",     op["add_effects"],    C["green_bg"], C["green_fg"], C["green_bd"])
        self._chip_row(detail, "−",     op["delete_effects"], C["red_bg"],   C["red_fg"],   C["red_bd"])

    def _chip_row(self, parent, label, items, bg, fg, bd):
        if not items:
            return
        row = tk.Frame(parent, bg=C["bg2"])
        row.pack(fill="x", pady=1)
        tk.Label(row, text=label, font=("Segoe UI", 8, "bold"),
                 fg=C["muted"], bg=C["bg2"], width=4, anchor="e").pack(side="left")
        for p in items:
            f = tk.Frame(row, bg=bd, padx=1, pady=1)
            f.pack(side="left", padx=(2, 0))
            tk.Label(f, text=p, font=("Courier New", 9),
                     bg=bg, fg=fg, padx=4, pady=1).pack()

    # ── Load config into UI ──────────────────────────────────────

    def _load_config_to_ui(self):
        self._initial_panel.set(self._config.get("initial_state", []))
        self._goal_panel.set(self._config.get("goal_predicates", []))
        self._refresh_operators_ui()

    # ── Run planner (called internally via ROS2 auto-run) ────────

    def _run_planner(self):
        initial = self._config.get("initial_state", [])
        goal    = self._config.get("goal_predicates", [])
        ops     = self._config.get("operators", [])
        depth   = 20

        if not initial or not goal or not ops:
            return

        print(f"[Planner] Running BFS  initial={initial}  goal={goal}  depth={depth}")
        self._write_result("Searching…\n", clear=True)

        def worker():
            t0 = time.perf_counter()
            plan, explored = plan_bfs(initial, goal, ops, depth)
            elapsed = (time.perf_counter() - t0) * 1000
            if plan is None:
                print(f"[Planner] No plan found  (explored {len(explored)} nodes, {elapsed:.1f} ms)")
            else:
                print(f"[Planner] Plan found!  {len(plan)} step(s)  |  {len(explored)} nodes explored  |  {elapsed:.1f} ms")
            self.after(0, lambda: self._show_result(plan, explored, elapsed, initial, goal, ops))

        threading.Thread(target=worker, daemon=True).start()

    def _show_result(self, plan, explored, elapsed_ms, initial, goal, ops):
        self._write_result("", clear=True)

        w = self._result_text
        w.configure(state="normal")

        if plan is None:
            w.insert("end", "No plan found\n", "err")
            w.insert("end", f"Searched {len(explored)} nodes within depth limit.\n", "info")
        else:
            w.insert("end", f"Plan found — {len(plan)} step{'s' if len(plan)!=1 else ''}\n", "ok")
            w.insert("end", f"Nodes explored: {len(explored)}   Time: {elapsed_ms:.1f} ms\n\n", "info")

            w.insert("end", "─── Plan ─────────────────────────────────\n", "head")
            for i, action in enumerate(plan, 1):
                w.insert("end", f"  {i:2d}. {action}\n", "step")

            w.insert("end", "\n─── State Trace ──────────────────────────\n", "head")
            trace = trace_plan(initial, plan, ops)
            goal_set = set(goal)
            for step in trace:
                w.insert("end", f"\nStep {step['step']:2d}  {step['action']}\n", "step")
                for p in sorted(step["state"]):
                    tag = "goal" if p in goal_set else "state"
                    marker = "  ✓ " if p in goal_set else "    "
                    w.insert("end", f"{marker}{p}\n", tag)

            final = set(trace[-1]["state"])
            if all(p in final for p in goal):
                w.insert("end", "\n✓ Goal state achieved\n", "ok")

        w.configure(state="disabled")
        w.see("1.0")

        if plan is not None:
            self._dispatch_plan(plan, ops)

    # ── Plan execution (send → wait → send next) ─────────────────

    def _dispatch_plan(self, plan: list, ops: list):
        if not self._ros2_bridge or not plan:
            return
        op_map = {o["name"]: o for o in ops}
        self._plan_execution = {
            "plan":            plan,
            "op_map":          op_map,
            "step":            0,
            "total":           len(plan),
            "waiting_for":     None,
            "collected_state": set(),
        }
        self._write_result("\n─── Execution ────────────────────────────\n")
        self._send_plan_step()

    def _send_plan_step(self):
        ex = self._plan_execution
        if ex is None:
            return
        step = ex["step"]
        if step >= ex["total"]:
            print(f"[Executor] All {ex['total']} operations complete")
            self._write_result(f"\n✓ All {ex['total']} operations confirmed.\n")
            self._plan_execution = None
            return
        action = ex["plan"][step]
        op = ex["op_map"][action]
        ex["waiting_for"] = set(op["add_effects"])
        ex["collected_state"] = set()          # reset accumulator for each new step

        # what /current_state must contain for this op to be considered done
        waiting_str = "  ".join(sorted(ex["waiting_for"]))
        print(f"[Executor] Sending step {step + 1}/{ex['total']}: {action}")
        print(f"[Executor]   waiting for /current_state to contain: {sorted(ex['waiting_for'])}")

        # preconditions the next operation will need (if there is one)
        next_step = step + 1
        if next_step < ex["total"]:
            next_action = ex["plan"][next_step]
            next_pre = ex["op_map"][next_action]["preconditions"]
            next_pre_str = "  ".join(sorted(next_pre))
            print(f"[Executor]   next op '{next_action}' needs: {sorted(next_pre)}")
        else:
            next_pre_str = None

        self._write_result(f"  → {step + 1:2d}. {action}\n")
        self._write_result(f"       complete when: {waiting_str}\n")
        if next_pre_str:
            self._write_result(f"       next op needs: {next_pre_str}\n")

        self._ros2_bridge.publish_command(action, step + 1)

    def receive_current_state(self, predicates: list[str]):
        print(f"[CurrentState] received: {sorted(predicates)}")

        ex = self._plan_execution
        if ex is None:
            return

        # Accumulate predicates across multiple messages for this step
        ex["collected_state"].update(predicates)
        collected = ex["collected_state"]
        print(f"[CurrentState] collected: {sorted(collected)}")

        waiting = ex.get("waiting_for")
        if waiting:
            matched   = waiting & collected
            unmatched = waiting - collected
            print(f"[CurrentState]   waiting : {sorted(waiting)}")
            print(f"[CurrentState]   matched : {sorted(matched)}")
            if unmatched:
                print(f"[CurrentState]   missing : {sorted(unmatched)}")

        if waiting and waiting.issubset(collected):
            step = ex["step"]
            action = ex["plan"][step]
            print(f"[Executor] Step {step + 1} confirmed ({action})")
            self._write_result(f"       ✓ {action} confirmed\n")
            ex["step"] += 1
            self._send_plan_step()

    # ── ROS2 integration ─────────────────────────────────────────

    def set_ros2_status(self, connected: bool,
                        initial_topic: str = "", goal_topic: str = ""):
        if connected:
            self._ros2_bar.pack(fill="x")
            self._ros2_dot.config(fg="#27a843")
            self._ros2_lbl.config(
                fg=C["text"],
                text=(f"ROS2 connected   "
                      f"initial → {initial_topic}   "
                      f"goal → {goal_topic}")
            )
        else:
            self._ros2_bar.pack_forget()

    def receive_initial_state(self, predicates: list[str]):
        print(f"[UI] Updated initial state ({len(predicates)} predicates): {predicates}")
        self._initial_panel.set(predicates)
        self._config["initial_state"] = predicates
        self._flash_panel(self._initial_panel, C["blue_bg"])
        self._ros2_dot.config(fg="#27a843")
        ts = time.strftime("%H:%M:%S")
        current = self._ros2_lbl.cget("text")
        base = current.split("  |  last")[0]
        self._ros2_lbl.config(text=f"{base}  |  last initial: {ts}")
        self._ros2_received["initial"] = True
        self._try_auto_run()

    def receive_goal_state(self, predicates: list[str]):
        print(f"[UI] Updated goal state ({len(predicates)} predicates): {predicates}")
        self._goal_panel.set(predicates)
        self._config["goal_predicates"] = predicates
        self._flash_panel(self._goal_panel, C["green_bg"])
        self._ros2_dot.config(fg="#27a843")
        ts = time.strftime("%H:%M:%S")
        current = self._ros2_lbl.cget("text")
        base = current.split("  |  last")[0]
        self._ros2_lbl.config(text=f"{base}  |  last goal: {ts}")
        self._ros2_received["goal"] = True
        self._try_auto_run()

    def _try_auto_run(self):
        if self._ros2_received.get("initial") and self._ros2_received.get("goal"):
            print("[Planner] Both states received — auto-running BFS...")
            self._run_planner()

    def _flash_panel(self, panel: "PredicatePanel", color: str):
        panel._chips_outer.config(bg=color)
        panel._chips_frame.config(bg=color)
        self.after(400, lambda: (
            panel._chips_outer.config(bg=C["bg2"]),
            panel._chips_frame.config(bg=C["bg2"]),
        ))

    # ── Helpers ──────────────────────────────────────────────────

    def _write_result(self, text: str, clear=False):
        self._result_text.configure(state="normal")
        if clear:
            self._result_text.delete("1.0", "end")
        if text:
            self._result_text.insert("end", text)
        self._result_text.configure(state="disabled")


# ─────────────────────────────────────────────────────────────────
# ROS2 Bridge
# ─────────────────────────────────────────────────────────────────

class Ros2Bridge:
    """
    Runs rclpy.spin() in a background daemon thread.
    Calls app.receive_initial_state() / app.receive_goal_state()
    on the Tkinter main thread via app.after(0, ...).
    """

    def __init__(self, app: "PlannerApp",
                 initial_topic:  str = "/task_planner/initial_state",
                 goal_topic:     str = "/task_planner/goal_state",
                 current_topic:  str = "/task_planner/current_state",
                 command_topic:  str = "/arm_command",
                 node_name:      str = "task_planner_ui"):

        if not _ROS2_AVAILABLE:
            raise RuntimeError(
                "rclpy is not installed. "
                "Install ROS2 and source setup.bash before using --ros2."
            )

        self._app = app
        import sys
        rclpy.init(args=sys.argv)
        self._node = Node(node_name)

        # ── Subscribers ───────────────────────────────────────────
        self._node.create_subscription(
            String, initial_topic,
            lambda msg: self._on_initial(msg), 10
        )
        self._node.create_subscription(
            String, goal_topic,
            lambda msg: self._on_goal(msg), 10
        )
        self._node.create_subscription(
            String, current_topic,
            lambda msg: self._on_current_state(msg), 10
        )

        # ── Publisher ─────────────────────────────────────────────
        self._cmd_pub = self._node.create_publisher(String, command_topic, 10)

        self._node.get_logger().info(
            f"Subscribed to:\n  initial → {initial_topic}\n"
            f"  goal    → {goal_topic}\n  current → {current_topic}\n"
            f"Publishing to: {command_topic}"
        )
        print(f"[ROS2] Node '{node_name}' started")
        print(f"[ROS2] Listening  initial_state : {initial_topic}")
        print(f"[ROS2] Listening  goal_state    : {goal_topic}")
        print(f"[ROS2] Listening  current_state : {current_topic}")
        print(f"[ROS2] Publishing arm_command   : {command_topic}")

        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def _spin(self):
        try:
            rclpy.spin(self._node)
        except Exception:
            pass

    def _parse(self, msg: "String") -> list[str] | None:
        try:
            data = json.loads(msg.data)
            if isinstance(data, list):
                return [str(p) for p in data]
            if isinstance(data, dict):
                preds = data.get("predicates", data.get("state", []))
                return [str(p) for p in preds]
        except json.JSONDecodeError:
            parts = [p.strip() for p in msg.data.split(",") if p.strip()]
            if parts:
                return parts
        return None

    def _on_initial(self, msg):
        preds = self._parse(msg)
        if preds is not None:
            print(f"[ROS2] Received initial_state: {preds}")
            self._app.after(0, lambda p=preds: self._app.receive_initial_state(p))
        else:
            print(f"[ROS2] Received initial_state but failed to parse: {msg.data!r}")

    def _on_goal(self, msg):
        preds = self._parse(msg)
        if preds is not None:
            print(f"[ROS2] Received goal_state:    {preds}")
            self._app.after(0, lambda p=preds: self._app.receive_goal_state(p))
        else:
            print(f"[ROS2] Received goal_state but failed to parse: {msg.data!r}")

    def _on_current_state(self, msg):
        preds = self._parse(msg)
        if preds is not None:
            self._app.after(0, lambda p=preds: self._app.receive_current_state(p))
        else:
            print(f"[ROS2] Received current_state but failed to parse: {msg.data!r}")

    # forward_mm: distance for move_to_pick's move_forward_and_align (millimetres)
    FORWARD_DIST_MM = {
        "move_to_pick": 100,
    }

    def publish_command(self, action: str, step: int):
        msg = String()
        payload = {"action": action, "step": step}
        if action in self.FORWARD_DIST_MM:
            payload["forward"] = self.FORWARD_DIST_MM[action]
        msg.data = json.dumps(payload)
        self._cmd_pub.publish(msg)
        print(f"[ROS2] Published  /arm_command: {msg.data}")

    def shutdown(self):
        self._node.destroy_node()
        rclpy.shutdown()


# ─────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="STRIPS Task Planner UI", add_help=False)
    p.add_argument("--ros2", action="store_true",
                   help="Enable ROS2 topic subscribers")
    p.add_argument("--initial-topic", default="/task_planner/initial_state",
                   help="ROS2 topic for initial state")
    p.add_argument("--goal-topic", default="/task_planner/goal_state",
                   help="ROS2 topic for goal state")
    p.add_argument("--node-name", default="task_planner_ui",
                   help="ROS2 node name")
    args, _ = p.parse_known_args()
    if _ROS2_AVAILABLE:
        args.ros2 = True
    return args


def main():
    args = _parse_args()

    print(f"[Planner] Starting  (ROS2 available: {_ROS2_AVAILABLE})")

    app = PlannerApp()

    bridge = None
    if args.ros2:
        if not _ROS2_AVAILABLE:
            print("[ERROR] rclpy not found. Source your ROS2 setup.bash and retry.")
        else:
            try:
                bridge = Ros2Bridge(
                    app,
                    initial_topic=args.initial_topic,
                    goal_topic=args.goal_topic,
                    node_name=args.node_name,
                )
                app._ros2_bridge = bridge
                app.set_ros2_status(
                    connected=True,
                    initial_topic=args.initial_topic,
                    goal_topic=args.goal_topic,
                )
            except Exception as e:
                print(f"[ERROR] ROS2 init failed: {e}")
    else:
        print("[Planner] ROS2 not enabled. Run with --ros2 or install rclpy.")

    def on_close():
        if bridge:
            bridge.shutdown()
        app.destroy()

    app.protocol("WM_DELETE_WINDOW", on_close)
    app.mainloop()


if __name__ == "__main__":
    main()
