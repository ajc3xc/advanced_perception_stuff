import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():

    # --- Build MoveIt configuration ---
    # MoveItConfigsBuilder reads all the config files from the
    # fairino10_v6_moveit2_config package automatically:
    # - fairino10_v6_robot.urdf.xacro  (robot description)
    # - fairino10_v6_robot.srdf         (planning groups)
    # - ros2_controllers.yaml           (controllers)
    # - moveit_controllers.yaml         (MoveIt controller bridge)
    # - kinematics.yaml                 (IK solver)
    # - joint_limits.yaml               (joint limits)
    # This is why we fixed those files in Step 2 — they all get loaded here.
    moveit_config = (
        MoveItConfigsBuilder(
            "fairino10_v6_robot",
            package_name="fairino10_v6_moveit2_config"
        )
        .planning_pipelines(default_planning_pipeline="ompl")
        .to_moveit_configs()
    )

    # --- move_group node ---
    # This is the core MoveIt node. It:
    # 1. Loads the robot model and planning scene
    # 2. Exposes planning services and action servers
    # 3. Connects to ros2_control controllers via moveit_controllers.yaml
    # 4. Publishes the planning scene for RViz to visualize
    #
    # We delay it by 8 seconds to ensure Gazebo has fully started
    # and the controllers are active before MoveIt tries to connect.
    move_group = TimerAction(
        period=8.0,
        actions=[
            Node(
                package='moveit_ros_move_group',
                executable='move_group',
                output='screen',
                parameters=[
                    moveit_config.to_dict(),
                    {'use_sim_time': True}
                ],
            )
        ]
    )

    # --- RViz ---
    # Visualization tool. We load it with the MoveIt RViz config
    # from the moveit2_config package so it has the Motion Planning
    # panel already set up.
    # We delay it by 9 seconds — it should start after move_group.
    rviz_config = os.path.join(
        get_package_share_directory('fairino10_v6_moveit2_config'),
        'config',
        'moveit.rviz'
    )

    rviz = TimerAction(
        period=9.0,
        actions=[
            Node(
                package='rviz2',
                executable='rviz2',
                output='screen',
                arguments=['-d', rviz_config],
                parameters=[
                    moveit_config.to_dict(),
                            {'use_sim_time': True}
                ],
            )
        ]
    )

    return LaunchDescription([
        move_group,
        rviz,
    ])