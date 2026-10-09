import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, RegisterEventHandler, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.event_handlers import OnProcessExit
from launch_ros.actions import Node
import xacro

def generate_launch_description():
    # 1. UPDATED: Package name changed to go2_description
    pkg_description = get_package_share_directory('go2_description')
    gazebo_resource_path = os.path.dirname(pkg_description)
    
    # 2. UPDATED: Controller YAML file name
    robot_controllers = os.path.join(pkg_description, 'config', 'go2_controllers.yaml')

    # 3. UPDATED: Xacro file path (assuming standard naming in your go2 package)
    xacro_file = os.path.join(pkg_description, 'xacro', 'robot.xacro')
    robot_description_config = xacro.process_file(xacro_file)
    robot_desc = {"robot_description": robot_description_config.toxml()}

    node_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[robot_desc, {'use_sim_time': True}]
    )
    
    pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim')
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={'gz_args': '-r empty.sdf'}.items(),
    )

    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-name', 'go2', 
            '-topic', 'robot_description',
            '-z', '0.37',
        ],
        output='screen',
    )

    load_joint_state_broadcaster = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_state_broadcaster",
            "-c", "/controller_manager",
            "--param-file", robot_controllers,
        ],
        parameters=[{'use_sim_time': True}]
    )

    load_state_estimator = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "state_estimator",
            "-c", "/controller_manager",
            "--param-file", robot_controllers,
        ],
        parameters=[{'use_sim_time': True}]
    )

    load_stand_up_controller = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "stand_up",
            "-c","/controller_manager",
            "--param-file", robot_controllers
        ],
        parameters=[{'use_sim_time': True}]
    )

    load_whole_body_controller = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "whole_body_controller",
            "-c", "/controller_manager",
            "--param-file", robot_controllers,
            "--inactive"
        ],
        parameters=[{'use_sim_time': True}]
    )

    clock_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            '/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry',
            '/imu@sensor_msgs/msg/Imu[gz.msgs.IMU',
        ],
        output='screen'
    )

    load_jsb_after_spawn = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=spawn_robot,
            on_exit=[load_joint_state_broadcaster],
        )
    )

    load_estimator_after_jsb = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=load_joint_state_broadcaster,
            on_exit=[load_state_estimator],
        )
    )

    load_stand_up_after_estimator = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=load_state_estimator,
            on_exit=[load_stand_up_controller],
        )
    )

    load_locomotion_after_stand_up = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=load_stand_up_controller,
            on_exit=[load_whole_body_controller],
        )
    )

    

    return LaunchDescription([
        SetEnvironmentVariable(name='GZ_SIM_RESOURCE_PATH',value=gazebo_resource_path),
        clock_bridge,
        node_robot_state_publisher,
        gazebo,
        spawn_robot,
        load_jsb_after_spawn,
        load_estimator_after_jsb,
        load_stand_up_after_estimator,
        load_locomotion_after_stand_up
    ])