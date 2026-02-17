
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource  # Import the missing source
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

print('file was updated')

def generate_launch_description():
    return LaunchDescription([
            IncludeLaunchDescription
            (
                PythonLaunchDescriptionSource([
                    PathJoinSubstitution([
                        FindPackageShare('clearpath_gz'), 'launch', 'simulation.launch.py'
                    ])
                ]),
                launch_arguments={
                    'world': 'orchard',
                    'rviz': 'true'
                }.items()
            )
    ])