import rclpy
from rclpy.node import Node
from geometry_msgs.msg import WrenchStamped
import tkinter as tk


class ForceInputNode(Node):

    def __init__(self):
        super().__init__('force_input_node')
        self.pub = self.create_publisher(WrenchStamped, '/force_sensor', 10)
        self.publishing = False

    def publish_force(self, entries):

        if not self.publishing:
            return

        msg = WrenchStamped()

        msg.wrench.force.x = float(entries[0].get())
        msg.wrench.force.y = float(entries[1].get())
        msg.wrench.force.z = float(entries[2].get())

        msg.wrench.torque.x = float(entries[3].get())
        msg.wrench.torque.y = float(entries[4].get())
        msg.wrench.torque.z = float(entries[5].get())

        self.pub.publish(msg)
        self.get_logger().info("Publishing Force")


def main():

    rclpy.init()
    node = ForceInputNode()

    root = tk.Tk()
    root.title("Force Sensor Input")

    # ขยายหน้าต่าง
    root.geometry("350x420")

    font_big = ("Arial", 16)

    labels = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]
    entries = []

    for i, l in enumerate(labels):
        tk.Label(root, text=l, font=font_big).grid(row=i, column=0, pady=8)

        e = tk.Entry(root, font=font_big, width=10)
        e.insert(0, "0")
        e.grid(row=i, column=1)
        entries.append(e)

    def toggle_publish():

        node.publishing = not node.publishing

        if node.publishing:
            btn.config(text="STOP", bg="gray")
            publish_loop()
        else:
            btn.config(text="START PUBLISH", bg="red")

    def publish_loop():

        if node.publishing:
            node.publish_force(entries)
            print(f"Force sent")
            root.after(50, publish_loop)   # publish 20 Hz

    btn = tk.Button(
        root,
        text="START PUBLISH",
        bg="red",
        fg="white",
        font=("Arial", 18),
        width=15,
        height=2,
        command=toggle_publish
    )

    btn.grid(row=7, column=0, columnspan=2, pady=30)

    root.mainloop()

    rclpy.shutdown()


if __name__ == '__main__':
    main()
