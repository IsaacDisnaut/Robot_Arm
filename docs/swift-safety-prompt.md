# AI Prompt: Robot Toolkit — Safety Status Buttons (Swift)

Copy the prompt below and give it to an AI (Claude, GPT, etc.) to generate the Swift code.

---

## Prompt

```
Create a Swift iOS/macOS view (SwiftUI) called `SafetyPanel` for a Robot Toolkit app.

## Purpose
This panel sends safety status commands to a robot arm via rosbridge v2 WebSocket protocol.
It publishes an Int8 value to the ROS topic `/safety_status`.

## Rosbridge Protocol
The robot uses rosbridge v2. All messages are JSON sent over a WebSocket connection.

WebSocket URL format: ws://<robotIP>:9090
Default IP: 192.168.1.100 (make this configurable via a TextField in the UI)

### Advertise topic on connect:
{
  "op": "advertise",
  "topic": "/safety_status",
  "type": "std_msgs/Int8"
}

### Publish message format:
{
  "op": "publish",
  "topic": "/safety_status",
  "msg": { "data": <value> }
}

Values:
- 0 = Normal (reset to safe state)
- 1 = Warning / Reduced Speed
- 2 = Emergency Stop

## UI Requirements

Layout: vertical stack, large touch-friendly buttons (min 80pt height), designed for iPad.

### Connection row (top):
- TextField for IP address (default: "192.168.1.100"), port fixed at 9090
- Connect / Disconnect button
- Small status indicator dot: gray=disconnected, green=connected, red=error

### Safety buttons (main content, 3 large buttons stacked):

1. **NORMAL / RESET** button
   - Color: green (#34C759 or similar)
   - Icon: checkmark.shield.fill
   - Label: "NORMAL"
   - Subtitle: "กลับสู่สถานะปกติ"
   - Publishes: { "data": 0 }

2. **WARNING** button
   - Color: orange (#FF9500)
   - Icon: exclamationmark.triangle.fill
   - Label: "WARNING"
   - Subtitle: "ลดความเร็ว / ระวัง"
   - Publishes: { "data": 1 }

3. **EMERGENCY STOP** button
   - Color: red (#FF3B30)
   - Icon: xmark.octagon.fill
   - Label: "EMERGENCY STOP"
   - Subtitle: "หยุดฉุกเฉิน"
   - Font: extra bold, larger than others
   - Add subtle pulse animation when active (current status = 2)
   - Publishes: { "data": 2 }

### Active state:
- Highlight the button that matches the last sent value with a ring/border or brighter background
- Show current status label below the buttons: "สถานะปัจจุบัน: ปกติ / เตือน / ฉุกเฉิน!"

### Feedback:
- Show a brief toast/overlay message after each button press: "ส่งคำสั่ง: EMERGENCY STOP" for 1.5 seconds

## Implementation Details

- Use `URLSessionWebSocketTask` for WebSocket (no external libraries)
- Reconnect automatically if connection drops (retry every 5 seconds)
- Disable safety buttons when not connected (gray them out)
- Handle JSON serialization with `JSONSerialization` or `Codable`
- Support both iOS 16+ and macOS 13+

## Code Structure

Provide:
1. `RosBridgeClient.swift` — WebSocket manager class (ObservableObject)
   - `connect(to url: URL)`
   - `disconnect()`
   - `publish(topic: String, type: String, data: [String: Any])`
   - `@Published var isConnected: Bool`

2. `SafetyPanel.swift` — SwiftUI view
   - Uses `@StateObject var ros = RosBridgeClient()`
   - All UI as described above

Make the code clean, well-commented in English, and production-ready.
```

---

## Expected Result

After generating, you will have two Swift files:
- `RosBridgeClient.swift` — reusable WebSocket manager for rosbridge v2
- `SafetyPanel.swift` — touch-friendly safety control panel

## Usage in Robot Toolkit

1. Add both files to your Robot Toolkit Xcode project
2. Add `SafetyPanel()` to your tab bar or navigation view
3. Set the robot IP to match your ROS machine
4. Press buttons to change safety status — the `robot-ui` badge will update in real time

## ROS Topic Reference

| Topic           | Type           | Values                        |
|-----------------|----------------|-------------------------------|
| `/safety_status`| std_msgs/Int8  | 0=ปกติ, 1=เตือน, 2=ฉุกเฉิน! |

The robot-ui web app subscribes to `/safety_status` and updates the safety badge automatically
whenever this topic receives a new value.
