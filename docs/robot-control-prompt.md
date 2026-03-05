# AI Prompt: Robot Controller Node (ROS2 Python)

Copy the prompt below and give it to an AI (Claude, GPT, etc.) to generate the robot-side ROS2 node.

---

## Prompt

```
Write a ROS2 Python node called `robot_controller` for a 6-axis robot arm with a linear rail.

## System Overview
This node connects to a web UI (FIBO Robot Cafe Studio) via rosbridge v2 WebSocket.
The UI sends motion commands via ROS topics. The node must execute them and report status back.

## Hardware
- 6-axis robot arm: joints J1–J6 (degrees)
- Linear rail: position in mm (0–600 mm)
- Gripper: position percentage (0–100%)
- All axes move simultaneously

## Topics to SUBSCRIBE (receive from UI)

### `/goto_position` — std_msgs/String (JSON)
Execute a single task. Parse msg.data as JSON.

**Joint mode payload (`controlMode: "joint"`):**
```json
{
  "sequence": 1,
  "label": "Pick cup",
  "j1": 45.0,
  "j2": -30.0,
  "j3": 90.0,
  "j4": 0.0,
  "j5": 45.0,
  "j6": 0.0,
  "rail": 200.0,
  "speed": 70,
  "gripper": 80,
  "controlMode": "joint"
}
```

**Effector mode payload (`controlMode: "effector"`):**
```json
{
  "sequence": 2,
  "label": "Place cup",
  "j1": 10.0,
  "j2": -20.0,
  "j3": 80.0,
  "j4": 5.0,
  "j5": 30.0,
  "j6": 0.0,
  "rail": 150.0,
  "speed": 50,
  "gripper": 60,
  "controlMode": "effector",
  "x": 312.5,
  "y": -45.2,
  "z": 380.0,
  "roll": 0.0,
  "pitch": -45.0,
  "yaw": 90.0
}
```

**Units:**
- x, y, z: millimeters (mm) in robot base frame
- roll, pitch, yaw: degrees
- rail: mm
- gripper: % (0–100)

**controlMode logic:**
- `"joint"` → Use joint-space interpolation. Move J1–J6 and rail directly to target angles. Fast, predictable path.
- `"effector"` → Use the Cartesian target (x, y, z, roll, pitch, yaw) for MoveIt2 IK planning.
  Convert units first: x/y/z mm → meters, roll/pitch/yaw degrees → radians.
  Call `move_group.set_pose_target(pose)` or `compute_cartesian_path`.
  The j1–j6 values in effector mode are provided as a hint/fallback only.

**speed field:** 1–100% of maximum joint velocity. Apply uniformly to all joints.

**Required behavior after receiving:**
1. Publish `/robot_status = 1` (executing)
2. Execute motion based on controlMode
3. Set gripper to target position
4. Publish `/robot_status = 0` (idle) when complete

### `/execute_trajectory` — std_msgs/String (JSON)
Full job sent at job start (for simulators). Schema:
```json
{
  "id": 3,
  "name": "Coffee Routine",
  "tasks": [
    { "sequence": 1, "label": "Home", "j1": 0, "j2": 0, "j3": 0, "j4": 0, "j5": 0, "j6": 0, "rail": 0, "speed": 50, "gripper": 0, "controlMode": "joint" },
    { "sequence": 2, "label": "Pick", "j1": 45, "j2": -30, "j3": 90, "j4": 0, "j5": 45, "j6": 0, "rail": 200, "speed": 70, "gripper": 80, "controlMode": "effector", "x": 312.5, "y": -45.2, "z": 380.0, "roll": 0.0, "pitch": -45.0, "yaw": 90.0 }
  ]
}
```
The UI executes tasks one at a time via /goto_position. This topic provides the full sequence upfront (optional use).

### `/pause_execution` — std_msgs/Bool
- `true` → pause current motion as safely as possible → publish `/robot_status = 2`
- `false` → resume motion → publish `/robot_status = 1` → publish `0` when done

### `/stop_execution` — std_msgs/Bool
- `true` → emergency stop. Decelerate immediately. Publish `/robot_status = 0`.

### `/teach_mode` — std_msgs/Bool
- `true` → enable free-drive (release joint brakes / reduce impedance)
- `false` → return to position hold

### `/tool_config` — std_msgs/String (JSON)
Configure the TCP (Tool Center Point) offset for IK planning. Parse msg.data:
```json
{
  "tcp_x": 0.0,
  "tcp_y": 0.0,
  "tcp_z": 150.0
}
```
- Values are in **millimeters**. Convert to meters before applying.
- Use `move_group.set_end_effector_link()` or update the tool frame offset so all future IK calls account for the physical tool tip position.
- Store in a node variable so it persists until the next `/tool_config` message.
- Publish acknowledgement to `/robot_status` is NOT required for this topic.

## Topics to PUBLISH (send to UI)

### `/joint_states` — sensor_msgs/JointState
Publish at 10 Hz. Report current positions for all axes:
```json
{
  "name": ["j1", "j2", "j3", "j4", "j5", "j6", "rail", "gripper"],
  "position": [j1_deg, j2_deg, j3_deg, j4_deg, j5_deg, j6_deg, rail_mm, gripper_pct],
  "velocity": [v1, v2, v3, v4, v5, v6]
}
```
- position[0–5]: joint angles in degrees
- position[6]: rail in mm
- position[7]: gripper in %
- velocity[0–5]: joint velocities in degrees/sec

### `/end_effector_pose` — std_msgs/String (JSON)
Publish at 10 Hz. Report current end-effector Cartesian pose, **accounting for the current TCP offset**:
```json
{
  "x": 312.5,
  "y": -45.2,
  "z": 380.0,
  "roll": 0.0,
  "pitch": -45.0,
  "yaw": 90.0
}
```
- x, y, z: **millimeters** in robot base frame (convert from MoveIt2 meters → mm)
- roll, pitch, yaw: **degrees** (convert from radians → degrees)
- Compute using `move_group.get_current_pose()` (which returns the tool frame if TCP offset is set)
- Publish even when the robot is idle (real-time feedback for the UI)

### `/robot_status` — std_msgs/Int8
Publish on every state change (not periodic):
- `0` = Idle (ready)
- `1` = Executing (moving)
- `2` = Paused (holding position)

### `/safety_status` — std_msgs/Int8
Publish when safety state changes:
- `0` = Normal
- `1` = Reduced speed / warning
- `2` = Emergency stop triggered

## Implementation Requirements

1. **ROS2 Humble or Jazzy** (Python rclpy)
2. **MoveIt2** for motion planning (use MoveGroupInterface)
3. Handle `controlMode` in separate methods:
   - `execute_joint_mode(task)` — set_joint_value_target for J1–J6 and rail
   - `execute_effector_mode(task)` — use x/y/z/roll/pitch/yaw from task payload
     - Convert: x, y, z (mm → m), roll, pitch, yaw (deg → rad)
     - Build `geometry_msgs/PoseStamped` with converted values
     - Call `move_group.set_pose_target(pose)` then `move_group.go(wait=True)`
     - Or use `compute_cartesian_path` for straight-line Cartesian movement
4. TCP offset handling:
   - Store tcp_x, tcp_y, tcp_z (in meters) from `/tool_config`
   - Apply as a static transform from the last joint frame to the tool frame
   - All subsequent `get_current_pose()` and IK calls must use this tool frame
5. Implement proper state machine for `/robot_status`:
   - IDLE → EXECUTING → IDLE (normal)
   - EXECUTING → PAUSED → EXECUTING (pause/resume)
   - any → IDLE (stop)
6. Thread-safe: motion execution in a background thread, subscribers on main thread
7. Publish `/joint_states` and `/end_effector_pose` in timer callbacks (10Hz each)

## Code Structure

Provide:
1. `robot_controller.py` — main ROS2 node
   - class `RobotController(Node)`
   - methods: `goto_position_cb`, `execute_joint_mode`, `execute_effector_mode`,
     `tool_config_cb`, `pause_cb`, `stop_cb`, `teach_mode_cb`,
     `joint_states_timer_cb`, `effector_pose_timer_cb`
2. Brief setup instructions for launching with MoveIt2

Make the code clean, well-commented in English, and production-ready for ROS2.
```

---

## Key Design Points

| Topic | Direction | Type | Purpose |
|-------|-----------|------|---------|
| `/goto_position` | UI → Robot | String/JSON | Execute one task (joint or effector mode) |
| `/execute_trajectory` | UI → Robot | String/JSON | Full job upfront (optional) |
| `/pause_execution` | UI → Robot | Bool | Pause (true) / Resume (false) |
| `/stop_execution` | UI → Robot | Bool | Emergency stop |
| `/teach_mode` | UI → Robot | Bool | Free-drive on/off |
| `/tool_config` | UI → Robot | String/JSON | TCP offset in mm (tcp_x, tcp_y, tcp_z) |
| `/joint_states` | Robot → UI | JointState | Live joint + rail + gripper feedback (10Hz) |
| `/end_effector_pose` | Robot → UI | String/JSON | Live Cartesian pose mm+deg (10Hz) |
| `/robot_status` | Robot → UI | Int8 | 0=idle, 1=executing, 2=paused |
| `/safety_status` | Robot → UI | Int8 | 0=normal, 1=warning, 2=emergency |

## controlMode Decision Flow

```
Receive /goto_position
        │
        ▼
controlMode == "joint" ?
   Yes ──► set_joint_value_target(j1..j6, rail)
            joint-space interpolation → move
   No  ──► convert x/y/z (mm→m), roll/pitch/yaw (deg→rad)
            build PoseStamped → set_pose_target / compute_cartesian_path
            Cartesian IK planning → move
        │
        ▼
Set gripper position
        │
        ▼
Publish /robot_status = 0
```

## TCP Offset Flow

```
Receive /tool_config  { tcp_x, tcp_y, tcp_z }  (mm)
        │
        ▼
Convert to meters, store as tool frame transform
        │
        ▼
All future get_current_pose() → returns tool tip position
All future set_pose_target()  → plans to tool tip position
        │
        ▼
/end_effector_pose reports tool tip pose (not flange)
```

## Unit Conventions

| Quantity | UI / DB | ROS topic | MoveIt2 internal |
|----------|---------|-----------|-----------------|
| Position (XYZ) | mm | mm (JSON string) | meters |
| Orientation | degrees | degrees (JSON string) | radians |
| Joint angles | degrees | degrees | degrees (joint mode) |
| Rail position | mm | mm | mm |
| Gripper | % | % | % |
