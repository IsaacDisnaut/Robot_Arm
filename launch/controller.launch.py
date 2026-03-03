from launch import LaunchDescription
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder
from ament_index_python.packages import get_package_share_directory
import os
def generate_launch_description():

    moveit_config = (
        MoveItConfigsBuilder("arctos_urdf", package_name="arctos_moveit_config")
        .robot_description(file_path=os.path.join(
            get_package_share_directory("robot_arm"),
            "urdf",
            "simple_rm_urdf.xacro"
            )
        )
        .planning_pipelines(pipelines=["ompl"])
        .robot_description_semantic(file_path="config/arctos_urdf.srdf")
        .trajectory_execution(file_path="config/moveit_controllers.yaml")
        .moveit_cpp(file_path="config/planning_python_api.yaml")
        .to_moveit_configs()
    )

    return LaunchDescription([
        Node(
            package="robot_arm",
            executable="mov_py",
            output="screen",
            parameters=[moveit_config.to_dict()]   # << สำคัญมาก
        )
    ])
