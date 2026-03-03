import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():

    # --- Build MoveIt configuration ---
    moveit_config = (
        MoveItConfigsBuilder(
            "fairino10_v6_robot",
            package_name="fairino10_v6_moveit2_config"
        )
        .planning_pipelines(default_planning_pipeline="ompl")
        .to_moveit_configs()
    )

    # --- move_group node ---
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