import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

class SimpleTrajectoryPublisher(Node):

    def __init__(self):
        super().__init__('simple_trajectory_publisher')

        self.publisher = self.create_publisher(
            JointTrajectory,
            '/arctos_controller/joint_trajectory',
            10
        )

        self.timer = self.create_timer(2.0, self.send_trajectory)

    def send_trajectory(self):

        msg = JointTrajectory()

        # 🔹 ใส่ชื่อ joint ให้ตรง URDF
        msg.joint_names = [
            'joint_1',
            'joint_2',
            'joint_3',
            'joint_4',
            'joint_5',
            'joint_6'
        ]

        point = JointTrajectoryPoint()

        # 🔹 ใส่ค่ามุมที่ต้องการ (rad)
        point.positions = [0.0, 0.5, 0.3, 0.0, 0.0, 0.0]

        # 🔹 เวลาที่จะให้ไปถึงตำแหน่งนี้
        point.time_from_start.sec = 2

        msg.points.append(point)

        self.publisher.publish(msg)
        self.get_logger().info("Sent trajectory")

def main(args=None):
    rclpy.init(args=args)
    node = SimpleTrajectoryPublisher()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
