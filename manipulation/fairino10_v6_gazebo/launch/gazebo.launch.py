import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
import xacro


def generate_launch_description():

    # --- Package paths ---
    # get_package_share_directory finds the installed location of a package.
    # This is why we need colcon build before launching — it looks in install/,
    # not in src/.
    gazebo_pkg = get_package_share_directory('fairino10_v6_gazebo')
    ros_gz_sim_pkg = get_package_share_directory('ros_gz_sim')

    # --- Process the URDF via xacro ---
    # xacro.process_file assembles our modular xacro files into one
    # complete URDF string that we pass to robot_state_publisher.
    urdf_file = os.path.join(gazebo_pkg, 'urdf', 'fairino10_v6_gazebo.urdf.xacro')
    robot_description_doc = xacro.process_file(urdf_file)
    robot_description = {'robot_description': robot_description_doc.toxml()}

    # --- 1. Start Gazebo Harmonic with our world ---
    # gz_sim.launch.py is provided by ros_gz_sim package.
    # We pass it our world file path via the gz_args argument.
    # -r means "run immediately" (don't pause on startup).
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim_pkg, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={
            'gz_args': os.path.join(gazebo_pkg, 'worlds', 'obstacles_world.sdf') + ' -r -s',
            'on_exit_shutdown': 'true'
        }.items()
    )

    # --- 2. Robot State Publisher ---
    # Reads the URDF and continuously publishes TF transforms for every
    # link in the robot. RViz, MoveIt, and Gazebo all rely on these
    # transforms to know where each part of the robot is in 3D space.
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
    # Gazebo Harmonic and ROS 2 use different communication systems.
    # Gazebo uses its own internal transport, ROS uses DDS.
    # This bridge connects them so they can share data.
    # We bridge the clock topic so ROS nodes use Gazebo's simulated time,
    # not the system wall clock. This is critical for accurate simulation.
    ros_gz_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
        ],
        output='screen'
    )

    # --- 4. Spawn the robot in Gazebo ---
    # This node tells Gazebo to create the robot model in the simulation.
    # It reads the robot_description topic (published by robot_state_publisher)
    # and spawns it at position x=0, y=0, z=0 (on the ground plane).
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
    # This ros2_control controller reads joint positions and velocities
    # from the hardware interface (Gazebo in our case) and publishes
    # them to /joint_states. MoveIt and robot_state_publisher both
    # subscribe to /joint_states to know the current robot configuration.
    # We delay it by 5 seconds to give Gazebo time to fully spawn the robot.
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
    # This is the JointTrajectoryController defined in ros2_controllers.yaml.
    # It receives trajectory commands from MoveIt via the
    # /fairino10_controller/follow_joint_trajectory action server
    # and sends position commands to each joint through the Gazebo bridge.
    # We delay it by 6 seconds — it must start after the broadcaster.
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