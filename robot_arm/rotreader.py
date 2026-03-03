import rclpy
from rclpy.node import Node
from tf2_ros import TransformException, Buffer, TransformListener
from tf_transformations import euler_from_quaternion
import math
import yaml

class RobotChainTracker(Node):
    def __init__(self):
        super().__init__('robot_chain_tracker')
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # กำหนดลำดับที่แน่นอนตามโครงสร้างหุ่นยนต์ของคุณ
        # ตรวจสอบชื่อ link ให้ตรงกับใน Rviz2 (เช่น link_5 หรือ joint_6)
        self.links = [
            'base_link', 'link_1', 'link_2', 'link_3', 
            'link_4', 'link_5', 'ee_link'
        ]
        
        self.timer = self.create_timer(1.0, self.on_timer)

    def on_timer(self):
        self.get_logger().info("--- Relative Orientations ---")
        
        for i in range(1, len(self.links)):
            child = self.links[i]
            parent = self.links[i-1]
            
            try:
                # ตรวจสอบว่ามี transform หรือไม่
                if self.tf_buffer.can_transform(parent, child, rclpy.time.Time()):
                    t = self.tf_buffer.lookup_transform(parent, child, rclpy.time.Time())
                    
                    q = t.transform.rotation
                    r, p, y = euler_from_quaternion([q.x, q.y, q.z, q.w])
                    
                    print(f"[{child:^10}] relative to [{parent:^10}] -> "
                          f"R: {r:>6.2f}° | "
                          f"P: {p:>6.2f}° | "
                          f"Y: {y:>6.2f}°")
                else:
                    self.get_logger().warn(f"Wait... No transform between {parent} and {child}")
                    
            except TransformException as e:
                continue

def main():
    rclpy.init()
    node = RobotChainTracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()