# Prompt สำหรับ AI: สร้าง Python RobotUIBridge

> Copy prompt ด้านล่างไปวางใน Claude / ChatGPT ได้เลย

---

## ── BEGIN PROMPT ──────────────────────────────────────────────────────────────

ฉันมี robot simulator เขียนด้วย Python (Robottoolkit หรืออื่นๆ) และต้องการเชื่อมต่อกับ
web UI ชื่อ **robot-ui** ผ่าน WebSocket

## ระบบ robot-ui ทำงานยังไง

- robot-ui เป็น Next.js web app สำหรับควบคุม 6-axis robotic arm + linear rail
- สื่อสารผ่าน **rosbridge v2 WebSocket protocol** ที่ port **9090**
- ตัวกลาง (relay broker) ชื่อ `mock-ros.ts` รันอยู่ที่ `ws://localhost:9090`
- UI ส่ง command → mock-ros relay → Python simulator
- Python simulator ตอบกลับ → mock-ros relay → UI

## rosbridge v2 Wire Format

ทุก message เป็น JSON object:

```json
// Subscribe (บอกว่าต้องการรับ topic นี้)
{ "op": "subscribe", "topic": "/topic_name", "type": "msg/Type" }

// Publish (ส่งข้อมูล)
{ "op": "publish", "topic": "/topic_name", "msg": { ...payload... } }
```

## Topics ที่ Python simulator ต้อง **รับ** (subscribe)

| Topic | Type | คำอธิบาย |
|-------|------|----------|
| `/goto_position` | std_msgs/String | UI สั่งให้หุ่นขยับไปตำแหน่งนั้น (ส่งที่ละ task) |
| `/execute_trajectory` | std_msgs/String | ส่ง job ทั้งหมด (ใช้เป็น reference เท่านั้น) |
| `/pause_execution` | std_msgs/Bool | `true`=pause, `false`=resume |
| `/stop_execution` | std_msgs/Bool | หยุดทันที |
| `/teach_mode` | std_msgs/Bool | `true`=free-drive mode |

### รูปแบบ `/goto_position` payload

`msg.data` เป็น JSON string ที่ต้อง parse อีกครั้ง:

```json
{
  "sequence": 1,
  "label": "Pick up cup",
  "j1": 45.0, "j2": -30.0, "j3": 90.0,
  "j4": 0.0, "j5": 45.0, "j6": 0.0,
  "rail": 200.0,
  "speed": 70,
  "gripper": 80
}
```

- `j1`–`j6`: degrees (−180 ถึง +180)
- `rail`: millimeters (0–600 mm)
- `speed`: percent (1–100%)
- `gripper`: percent (0=ปิด, 100=เปิดสุด)

## Topics ที่ Python simulator ต้อง **ส่ง** (publish)

### `/robot_status` — สำคัญมาก!

UI รอฟัง topic นี้ก่อนส่ง task ถัดไป ต้องส่งตรงเวลา:

```json
{ "op": "publish", "topic": "/robot_status", "msg": { "data": 0 } }
```

| ค่า | ความหมาย | เมื่อไหร่ |
|-----|----------|----------|
| `0` | Idle — พร้อมรับ command | หลังถึง target / หลัง stop |
| `1` | Executing — กำลังเคลื่อนที่ | ทันทีที่รับ `/goto_position` |
| `2` | Paused | เมื่อรับ `/pause_execution true` |

**State machine ที่สำคัญ:**
```
รับ /goto_position  →  ส่ง status=1  →  เคลื่อนที่  →  ส่ง status=0
รับ /pause_execution true   →  ส่ง status=2  (hold position)
รับ /pause_execution false  →  ส่ง status=1  →  resume  →  ส่ง status=0
รับ /stop_execution         →  ส่ง status=0  ทันที
```

### `/joint_states` — 10 Hz

```json
{
  "op": "publish",
  "topic": "/joint_states",
  "msg": {
    "name": ["j1","j2","j3","j4","j5","j6","rail","gripper"],
    "position": [0.0, 45.0, -30.0, 0.0, 90.0, 0.0, 150.0, 80.0],
    "velocity": [0.0, 12.5, -8.3, 0.0, 0.0, 0.0]
  }
}
```

- `position[0–5]`: J1–J6 in degrees
- `position[6]`: rail in mm
- `position[7]`: gripper %
- `velocity[0–5]`: degrees/sec (6 values)

### `/safety_status` — optional, ส่งเมื่อเปลี่ยน

```json
{ "op": "publish", "topic": "/safety_status", "msg": { "data": 0 } }
```
`0`=normal, `1`=reduced speed, `2`=emergency stop

## สิ่งที่ฉันต้องการให้สร้าง

สร้าง Python class ชื่อ **`RobotUIBridge`** ที่:

1. Connect WebSocket ไปที่ `ws://localhost:9090`
2. Subscribe topics ข้างต้นทันทีที่ connect
3. รับ callback hooks ให้ฉัน bind กับ code ที่มีอยู่:

```python
bridge = RobotUIBridge(url="ws://localhost:9090")

# Bind กับ robot controller ที่มีอยู่
bridge.on_goto_position = lambda task: my_robot.move_to(task)
bridge.on_pause = lambda: my_robot.pause()
bridge.on_resume = lambda: my_robot.resume()
bridge.on_stop = lambda: my_robot.stop()
bridge.on_teach_mode = lambda enabled: my_robot.set_freedrive(enabled)

# อัปเดต joint states real-time (เรียกจาก loop ของฉัน)
bridge.update_joints(
    joints=[j1, j2, j3, j4, j5, j6],  # degrees
    rail=rail_mm,
    gripper=gripper_pct,
    velocities=[v1, v2, v3, v4, v5, v6]
)

# อัปเดต robot status
bridge.set_status(0)   # 0=idle, 1=executing, 2=paused

bridge.start()  # blocking loop (หรือ start_async() สำหรับ async)
```

## Requirements สำคัญ

1. **Publish `status=1` ก่อนขยับเสมอ** — UI รอสูงสุด 600ms ก่อนจะถือว่า simulator ไม่ respond
2. **Publish `status=0` ทันทีที่ถึง target** — UI รอ status=0 ก่อนส่ง task ถัดไป
3. **Pause/Resume ต้องถูกต้อง:**
   - รับ `pause_execution=true` → ส่ง `status=2` → hold ไม่ขยับ
   - รับ `pause_execution=false` → ส่ง `status=1` → ขยับต่อ → ส่ง `status=0` เมื่อเสร็จ
4. **`delay` field ใน goto_position ไม่ต้องทำ** — UI จัดการ delay เอง
5. **Joints ทุกตัวเคลื่อนที่พร้อมกัน** (interpolate พร้อมกัน ไม่ใช่ทีละ joint)
6. **ส่ง `/joint_states` ที่ 10 Hz** ตลอดเวลา (ไม่ใช่แค่ตอนเคลื่อนที่)
7. **`msg.data` ใน `/goto_position` เป็น JSON string** ต้อง `json.loads()` อีกรอบ

## ตัวอย่าง integration กับ code ที่มีอยู่

```python
import my_robot_controller as robot

bridge = RobotUIBridge()

def on_goto(task: dict):
    # task มี: j1,j2,j3,j4,j5,j6 (degrees), rail (mm), speed (%), gripper (%)
    bridge.set_status(1)
    robot.move_joints(
        joints=[task['j1'], task['j2'], task['j3'],
                task['j4'], task['j5'], task['j6']],
        rail=task['rail'],
        speed=task['speed'] / 100.0,
        gripper=task['gripper']
    )
    # หลังขยับเสร็จ:
    bridge.set_status(0)

bridge.on_goto_position = on_goto
bridge.on_stop = lambda: robot.emergency_stop()
bridge.on_pause = lambda: robot.pause()
bridge.on_resume = lambda: robot.resume()

# Thread/loop ที่ publish joint states:
def joint_publisher():
    while True:
        j = robot.get_joint_positions()
        bridge.update_joints(j.angles, j.rail, j.gripper, j.velocities)
        time.sleep(0.1)  # 10 Hz

threading.Thread(target=joint_publisher, daemon=True).start()
bridge.start()
```

## Dependencies ที่ยอมรับได้

- `websocket-client` หรือ `websockets` (async)
- `threading` หรือ `asyncio`
- stdlib เท่านั้น ถ้าเป็นไปได้

กรุณาสร้าง `RobotUIBridge` ที่สมบูรณ์พร้อม docstring และตัวอย่างการใช้งาน

## ── END PROMPT ────────────────────────────────────────────────────────────────
