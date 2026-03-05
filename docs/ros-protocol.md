# Robot ROS Interface Protocol

## Overview

This document defines the complete ROS communication protocol between the **robot-ui** web application and the robot controller. It is intended as a specification for the robot-side AI/engineer to implement.

The UI communicates with the robot via **rosbridge v2** WebSocket protocol running on **port 9090**.

During development, `scripts/mock-ros.ts` acts as a relay broker. In production, replace it with a real rosbridge server running on the robot.

---

## Units

| Axis          | Unit        | Range           |
|---------------|-------------|-----------------|
| J1 – J6       | degrees (°) | -180 to +180    |
| Linear Rail   | millimeters | 0 to 600 mm     |
| Gripper       | percent (%) | 0 to 100        |
| Speed         | percent (%) | 1 to 100        |
| Delay         | milliseconds| 0 to 60000      |
| Velocity      | degrees/sec | 0 to ~200 °/s   |

---

## Topics: Robot → UI (Robot Publishes)

### `/joint_states`
**Message type:** `sensor_msgs/JointState`
**Rate:** 10 Hz (100ms interval)

Publishes current joint positions, velocities, and gripper state.

```json
{
  "op": "publish",
  "topic": "/joint_states",
  "msg": {
    "name": ["j1", "j2", "j3", "j4", "j5", "j6", "rail", "gripper"],
    "position": [0.0, 45.0, -30.0, 0.0, 90.0, 0.0, 150.0, 80.0],
    "velocity": [0.0, 12.5, -8.3, 0.0, 0.0, 0.0]
  }
}
```

- `position[0-5]` → J1–J6 in degrees
- `position[6]` → Rail position in mm
- `position[7]` → Gripper open % (0=closed, 100=fully open)
- `velocity[0-5]` → Joint velocities in degrees/sec (6 values)

---

### `/robot_status`
**Message type:** `std_msgs/Int8`
**Rate:** publish on every state change (not periodic)

Reports the current execution state of the robot.

```json
{ "op": "publish", "topic": "/robot_status", "msg": { "data": 0 } }
```

| Value | State     | Description                                               |
|-------|-----------|-----------------------------------------------------------|
| `0`   | Idle      | Robot is stationary, ready to accept commands             |
| `1`   | Executing | Robot is currently moving toward a target position        |
| `2`   | Paused    | Robot has paused mid-trajectory (received /pause_execution) |

**Required state transitions:**
- After receiving `/goto_position`: publish `{ "data": 1 }` → move → publish `{ "data": 0 }`
- After receiving `/pause_execution { data: true }`: publish `{ "data": 2 }` (hold position)
- After receiving `/pause_execution { data: false }` while paused: publish `{ "data": 1 }` → resume → publish `{ "data": 0 }`
- After receiving `/stop_execution`: publish `{ "data": 0 }` immediately

---

### `/safety_status`
**Message type:** `std_msgs/Int8`
**Rate:** publish on every change

Reports safety/speed-limit state.

```json
{ "op": "publish", "topic": "/safety_status", "msg": { "data": 0 } }
```

| Value | State          | UI Effect                     |
|-------|----------------|-------------------------------|
| `0`   | Normal         | Status badge: green           |
| `1`   | Reduced Speed  | Status badge: orange + warning|
| `2`   | Emergency Stop | Status badge: red + halt UI   |

---

## Topics: UI → Robot (Robot Subscribes)

### `/goto_position`
**Message type:** `std_msgs/String`
**Data:** JSON string (see below)

Sent for each individual task during job execution. The UI sends tasks **one at a time** and waits for `/robot_status=0` before sending the next.

```json
{
  "op": "publish",
  "topic": "/goto_position",
  "msg": {
    "data": "{\"sequence\":1,\"label\":\"Pick up cup\",\"j1\":45.0,\"j2\":-30.0,\"j3\":90.0,\"j4\":0.0,\"j5\":45.0,\"j6\":0.0,\"rail\":200.0,\"speed\":70,\"gripper\":80}"
  }
}
```

**Payload fields (inside `msg.data` as JSON string):**

| Field         | Type   | Description                                              |
|---------------|--------|----------------------------------------------------------|
| `sequence`    | int    | Task number (1-based)                                    |
| `label`       | string | Human-readable task name                                 |
| `j1`–`j6`    | float  | Target joint angles in degrees                           |
| `rail`        | float  | Target rail position in mm                               |
| `speed`       | int    | Movement speed (1–100%)                                  |
| `gripper`     | int    | Target gripper position (0–100%)                         |
| `controlMode` | string | `"joint"` (default) or `"effector"` — motion planner    |

**`controlMode` behavior:**
- `"joint"` → Move each joint directly to the target angle (joint-space interpolation). Fast and predictable.
- `"effector"` → Compute FK from j1–j6 to get the Cartesian target pose, then use Cartesian/task-space motion planning (e.g., MoveIt! Cartesian path). The end-effector follows a straight line in Cartesian space.

**Robot behavior:**
1. Parse `msg.data` as JSON
2. Check `controlMode` to select motion planner
3. Publish `/robot_status = 1`
4. Execute motion (joint interpolation or Cartesian planning)
5. Set gripper to target value
6. Publish `/robot_status = 0` when all axes have reached their targets

> **Note:** The `delay` field is **not** included in `/goto_position`. Delay is handled entirely by the UI (countdown shown between tasks). The robot should simply move and report done.

---

### `/execute_trajectory`
**Message type:** `std_msgs/String`
**Data:** JSON string (full job)

Sent when the user clicks "เริ่มทำงาน" (Start Job). Contains the entire job with all tasks. Use this topic if you want the robot to execute the full sequence autonomously without waiting for UI commands per task.

```json
{
  "op": "publish",
  "topic": "/execute_trajectory",
  "msg": {
    "data": "{\"id\":3,\"name\":\"Coffee Routine\",\"tasks\":[{\"sequence\":1,\"label\":\"Home\",\"j1\":0,\"j2\":0,\"j3\":0,\"j4\":0,\"j5\":0,\"j6\":0,\"rail\":0,\"speed\":50,\"delay\":0,\"gripper\":0},{\"sequence\":2,\"label\":\"Pick\",\"j1\":45,\"j2\":-30,\"j3\":90,\"j4\":0,\"j5\":45,\"j6\":0,\"rail\":200,\"speed\":70,\"delay\":2000,\"gripper\":80}]}"
  }
}
```

**Full payload schema:**
```json
{
  "id": 3,
  "name": "Job name",
  "tasks": [
    {
      "sequence": 1,
      "label": "Task label",
      "j1": 0.0, "j2": 0.0, "j3": 0.0,
      "j4": 0.0, "j5": 0.0, "j6": 0.0,
      "rail": 0.0,
      "speed": 50,
      "delay": 2000,
      "gripper": 0
    }
  ]
}
```

> **Current UI behavior:** The UI sends tasks one at a time via `/goto_position`. `/execute_trajectory` is also published at job start for simulators that want the full sequence upfront.

---

### `/pause_execution`
**Message type:** `std_msgs/Bool`

Pause or resume robot motion.

```json
// Pause
{ "op": "publish", "topic": "/pause_execution", "msg": { "data": true } }

// Resume
{ "op": "publish", "topic": "/pause_execution", "msg": { "data": false } }
```

**Robot behavior:**
- On `true`: hold current position as safely as possible → publish `/robot_status = 2`
- On `false` while paused: resume motion toward last target → publish `/robot_status = 1` → publish `/robot_status = 0` when done

---

### `/stop_execution`
**Message type:** `std_msgs/Bool`

Emergency stop — halt immediately, cancel current trajectory.

```json
{ "op": "publish", "topic": "/stop_execution", "msg": { "data": true } }
```

**Robot behavior:**
1. Decelerate and stop as quickly as safely possible
2. Cancel any queued movements
3. Publish `/robot_status = 0`

---

### `/teach_mode`
**Message type:** `std_msgs/Bool`

Enable/disable teach mode (free-drive / back-driving).

```json
// Enable teach mode
{ "op": "publish", "topic": "/teach_mode", "msg": { "data": true } }

// Disable teach mode
{ "op": "publish", "topic": "/teach_mode", "msg": { "data": false } }
```

**Robot behavior:**
- On `true`: release joint brakes / reduce impedance → allow manual positioning
- On `false`: re-engage position hold

---

## Complete Topic Summary

| Topic                 | Direction   | Type                  | Description                        |
|-----------------------|-------------|-----------------------|------------------------------------|
| `/joint_states`       | Robot → UI  | sensor_msgs/JointState| Current positions + velocities     |
| `/robot_status`       | Robot → UI  | std_msgs/Int8         | 0=idle, 1=executing, 2=paused      |
| `/safety_status`      | Robot → UI  | std_msgs/Int8         | 0=normal, 1=slow, 2=emergency      |
| `/goto_position`      | UI → Robot  | std_msgs/String (JSON)| Move to single task position       |
| `/execute_trajectory` | UI → Robot  | std_msgs/String (JSON)| Full job sequence (all tasks)      |
| `/pause_execution`    | UI → Robot  | std_msgs/Bool         | true=pause, false=resume           |
| `/stop_execution`     | UI → Robot  | std_msgs/Bool         | Emergency stop                     |
| `/teach_mode`         | UI → Robot  | std_msgs/Bool         | true=free-drive, false=position-hold|

---

## Execution Flow

```
UI                              Robot
 │                                │
 │── /goto_position (task 1) ────►│
 │                                │── publish /robot_status = 1
 │                                │   (moving...)
 │                                │── publish /robot_status = 0
 │◄────────────────────────────── │
 │   (UI starts delay countdown)  │
 │── /goto_position (task 2) ────►│
 │                                │── publish /robot_status = 1
 │     [user presses Pause]       │
 │── /pause_execution true ──────►│
 │                                │── publish /robot_status = 2
 │     [user presses Resume]      │
 │── /pause_execution false ─────►│
 │                                │── publish /robot_status = 1
 │                                │   (resumes moving...)
 │                                │── publish /robot_status = 0
 │◄────────────────────────────── │
```

---

## IK Simulator Integration

For the IK visual simulator to integrate with robot-ui via `scripts/mock-ros.ts`:

**On WebSocket open:**
```javascript
ws.send(JSON.stringify({ "op": "subscribe", "topic": "/goto_position",      "type": "std_msgs/String" }));
ws.send(JSON.stringify({ "op": "subscribe", "topic": "/execute_trajectory", "type": "std_msgs/String" }));
ws.send(JSON.stringify({ "op": "subscribe", "topic": "/pause_execution",    "type": "std_msgs/Bool" }));
ws.send(JSON.stringify({ "op": "subscribe", "topic": "/stop_execution",     "type": "std_msgs/Bool" }));
```

**Publishing joint states (100ms interval):**
```javascript
ws.send(JSON.stringify({
  "op": "publish",
  "topic": "/joint_states",
  "msg": {
    "name": ["j1","j2","j3","j4","j5","j6","rail","gripper"],
    "position": [j1, j2, j3, j4, j5, j6, rail, gripper],
    "velocity": [v1, v2, v3, v4, v5, v6]
  }
}));
```

**Publishing robot status:**
```javascript
// When starting motion
ws.send(JSON.stringify({ "op": "publish", "topic": "/robot_status", "msg": { "data": 1 } }));

// When motion complete
ws.send(JSON.stringify({ "op": "publish", "topic": "/robot_status", "msg": { "data": 0 } }));

// When paused
ws.send(JSON.stringify({ "op": "publish", "topic": "/robot_status", "msg": { "data": 2 } }));
```

---

## Notes for Robot-Side Developer

1. **Always publish `/robot_status`** — The UI progress bar and task sequencing depend entirely on these status updates.
2. **Publish `/robot_status = 1` before starting motion**, not after — The UI waits up to 600ms for motion to start.
3. **`delay` is UI-only** — Do not implement delay on the robot side. After reaching the target, publish `status=0` immediately.
4. **Gripper is position-based** — `gripper: 80` means 80% open. Implement as absolute position, not toggle.
5. **Speed scaling** — `speed: 50` means 50% of maximum velocity. Apply uniformly to all axes of that task.
6. **All axes move simultaneously** — Do not move joints sequentially; all J1–J6 and rail should start and arrive together.
