import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution

from launch_ros.actions import Node


def generate_launch_description():
    package_path = get_package_share_directory('ellipselio')
    default_config_path = os.path.join(package_path, 'config')
    default_rviz_config_path = os.path.join(
        package_path, 'rviz', 'ellipselio.rviz')

    use_sim_time = LaunchConfiguration('use_sim_time')
    config_path = LaunchConfiguration('config_path')
    config_file = LaunchConfiguration('config_file')
    rviz_use = LaunchConfiguration('rviz')
    rviz_cfg = LaunchConfiguration('rviz_cfg')
    container_name = LaunchConfiguration('container_name')

    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time', default_value='true',
        description='Use simulation clock if true'
    )
    declare_config_path_cmd = DeclareLaunchArgument(
        'config_path', default_value=default_config_path,
        description='Yaml config file path'
    )
    declare_config_file_cmd = DeclareLaunchArgument(
        'config_file', default_value='qt64_spires.yaml',
        description='Config file'
    )
    declare_rviz_cmd = DeclareLaunchArgument(
        'rviz', default_value='true',
        description='Use RViz to monitor results'
    )
    declare_rviz_config_path_cmd = DeclareLaunchArgument(
        'rviz_cfg', default_value=default_rviz_config_path,
        description='RViz config file path'
    )
    container_name_arg = DeclareLaunchArgument(
        name='container_name',
        default_value='ellipselio_container',
        description='container name')

    declare_bag_path_cmd = DeclareLaunchArgument(
        'bag_path', default_value='',
        description='Bag being played; results go to <bag parent>/lio_res/ellipselio'
    )
    declare_save_path_cmd = DeclareLaunchArgument(
        'save_path', default_value='',
        description='Explicit output directory; overrides bag_path. Empty disables saving'
    )
    declare_save_scans_cmd = DeclareLaunchArgument(
        'save_scans', default_value='true',
        description='Dump each frame to save_path/scans/NNNNNN.pcd'
    )
    declare_save_scans_local_cmd = DeclareLaunchArgument(
        'save_scans_local', default_value='true',
        description='Per-scan clouds in the LiDAR frame (true, for BA) or world frame'
    )

    container = Node(
        package='rclcpp_components',
        executable='component_container_mt',
        name=container_name,
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
    )

    base_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([package_path, 'launch', 'ellipselio.launch.py'])
        ),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'config_path': config_path,
            'config_file': config_file,
            'rviz': rviz_use,
            'rviz_cfg': rviz_cfg,
            'container_name': container_name,
            'bag_path': LaunchConfiguration('bag_path'),
            'save_path': LaunchConfiguration('save_path'),
            'save_scans': LaunchConfiguration('save_scans'),
            'save_scans_local': LaunchConfiguration('save_scans_local'),
        }.items(),
    )

    ld = LaunchDescription()
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_config_path_cmd)
    ld.add_action(declare_config_file_cmd)
    ld.add_action(declare_rviz_cmd)
    ld.add_action(declare_rviz_config_path_cmd)
    ld.add_action(container_name_arg)
    ld.add_action(declare_bag_path_cmd)
    ld.add_action(declare_save_path_cmd)
    ld.add_action(declare_save_scans_cmd)
    ld.add_action(declare_save_scans_local_cmd)
    ld.add_action(container)
    ld.add_action(base_launch)

    return ld
