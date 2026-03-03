import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
import xacro


def generate_launch_description():

    # --- Package paths ---
    gazebo_pkg = get_package_share_directory('fairino10_v6_gazebo')
    ros_gz_sim_pkg = get_package_share_directory('ros_gz_sim')

    # --- Process the URDF via xacro ---
    urdf_file = os.path.join(gazebo_pkg, 'urdf', 'fairino10_v6_gazebo.urdf.xacro')
    robot_description_doc = xacro.process_file(urdf_file)
    robot_description = {'robot_description': robot_description_doc.toxml()}

    # --- 1. Start Gazebo Harmonic with our world ---
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim_pkg, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={
            'gz_args': os.path.join(gazebo_pkg, 'worlds', 'obstacles_world.sdf') + ' -r ',
            'on_exit_shutdown': 'true'
        }.items()
    )

    # --- 2. Robot State Publisher ---
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[
            robot_description,
            {'use_sim_time': True}
        ]  # Use Gazebo's simulated time
    )

    # --- 3. ROS-Gazebo Bridge ---
    ros_gz_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
        ],
        output='screen'
    )

    # --- 4. Spawn the robot in Gazebo ---
    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-name', 'fairino10_v6',
            '-topic', 'robot_description',
            '-x', '0.0',
            '-y', '0.0',
            '-z', '0.0',
        ],
        output='screen'
    )

    # --- 5. Joint State Broadcaster ---
    joint_state_broadcaster = TimerAction(
        period=5.0,
        actions=[
            Node(
                package='controller_manager',
                executable='spawner',
                arguments=['joint_state_broadcaster'],
                output='screen',
            )
        ]
    )

    # --- 6. Arm Trajectory Controller ---
    arm_controller = TimerAction(
        period=6.0,
        actions=[
            Node(
                package='controller_manager',
                executable='spawner',
                arguments=['fairino10_controller'],
                output='screen',
            )
        ]
    )

    return LaunchDescription([
        gazebo,
        robot_state_publisher,
        ros_gz_bridge,
        spawn_robot,
        joint_state_broadcaster,
        arm_controller,
    ])