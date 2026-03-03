import os
from launch import LaunchDescription
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder
from ament_index_python.packages import get_package_share_directory
import yaml


def generate_launch_description():

    moveit_config = (
        MoveItConfigsBuilder(
            "fairino10_v6_robot",
            package_name="fairino10_v6_moveit2_config"
        )
        .planning_pipelines(default_planning_pipeline="ompl")
        .trajectory_execution(file_path="config/moveit_controllers.yaml")
        .to_moveit_configs()
    )

    moveit_py_yaml_path = os.path.join(
        get_package_share_directory('fairino10_v6_moveit2_config'),
        'config',
        'moveit_py.yaml'
    )
    with open(moveit_py_yaml_path, 'r') as f:
        moveit_py_params = yaml.safe_load(f)

    planning_node = Node(
        package='fairino10_v6_gazebo',
        executable='planning_node',
        output='screen',
        parameters=[
            {'use_sim_time': True},
            moveit_config.to_dict(),
            moveit_py_params,
            {'trajectory_execution.execution_duration_monitoring': False},
        ],
    )

    return LaunchDescription([planning_node])