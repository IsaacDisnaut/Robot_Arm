from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():

    rviz_config = '/home/isaac/ros2_ws/src/robot_arm/rviz/ik.rviz'

    urdf = os.path.join(
        get_package_share_directory('robot_arm'),
        'urdf',
        'simple_rm_urdf.xacro'
    )

    return LaunchDescription([
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': open(urdf).read()}]
        ),

        # Node(
        #     package='joint_state_publisher_gui',
        #     executable='joint_state_publisher_gui',
        #     output='screen'
        # ),
        Node(
            package='robot_arm',
            executable='IK',
            output='screen'
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', rviz_config]
        )
    ])