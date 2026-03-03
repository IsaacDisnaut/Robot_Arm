import os
from launch import LaunchDescription
from moveit_configs_utils import MoveItConfigsBuilder
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument,ExecuteProcess
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

    is_sim = LaunchConfiguration("is_sim")
    
    is_sim_arg = DeclareLaunchArgument(
        "is_sim",
        default_value="True"
    )
    moveit_config = (
        MoveItConfigsBuilder("arctos", package_name="robot_arm")
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

    simple_moveit_interface = Node(
        package="robot_arm",
        executable="mov_py",
        parameters=[moveit_config.to_dict(),{"use_sim_time":True}]
    
    )

        # Robot Description
    robot_description = moveit_config.robot_description

    # ros2_control node
    ros2_control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[
            moveit_config.robot_description, 
            os.path.join(
                get_package_share_directory("arctos_moveit_config"),
                "config",
                "ros2_controllers.yaml",
            ),
        ],
        output="screen",
    )

    spawn_jsb = Node(
    package="controller_manager",
    executable="spawner",
    arguments=["joint_state_broadcaster"],
)

    # Spawn arm controller
    spawn_arm = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["arm_controller"],
        output="screen",
    )

    # Spawn gripper controller
    spawn_gripper = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["gripper_controller"],
        output="screen",
    )
    robot_state_publisher = Node(
    package="robot_state_publisher",
    executable="robot_state_publisher",
    output="screen",
    parameters=[moveit_config.robot_description],
)


    return LaunchDescription([
        is_sim_arg,
        ros2_control_node,
        spawn_arm,
        spawn_jsb,  
        spawn_gripper,
        simple_moveit_interface,
        robot_state_publisher,
    ])
