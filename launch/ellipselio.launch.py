import os.path

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, SetLaunchConfiguration
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.conditions import IfCondition

from launch_ros.actions import ComposableNodeContainer, Node, LoadComposableNodes
from launch_ros.descriptions import ComposableNode


def resolve_save_path(context):
    """Derive save_path from bag_path when it was not given explicitly.

    A bag at <dataset>/<bag_name> yields <dataset>/lio_res/ellipselio, i.e. the
    results land in a sibling of the bag directory.
    """
    save_path = LaunchConfiguration('save_path').perform(context)
    if save_path:
        return [SetLaunchConfiguration('save_path', save_path)]

    bag_path = LaunchConfiguration('bag_path').perform(context)
    if not bag_path:
        return []

    bag = os.path.abspath(os.path.expanduser(bag_path.rstrip('/')))
    derived = os.path.join(os.path.dirname(bag), 'lio_res', 'ellipselio')
    return [SetLaunchConfiguration('save_path', derived)]

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
        default_value="ellipselio_container",
        description="container name")

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

    save_params = {
        'mapping.save_path': LaunchConfiguration('save_path'),
        'mapping.save_scans': LaunchConfiguration('save_scans'),
        'mapping.save_scans_local': LaunchConfiguration('save_scans_local'),
    }

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', rviz_cfg],
        condition=IfCondition(rviz_use),
        output='log'
    )

    ellipse_lio_node = Node(
        package='ellipselio',
        executable='ellipselio_mapping_node',
        name='ellipselio',
        parameters=[PathJoinSubstitution([config_path, config_file]),
                    {'use_sim_time': use_sim_time},
                    save_params],
        output='screen',
        prefix=['gdbserver localhost:3000'],
    )

    ellipse_lio_comp = ComposableNode(
        package='ellipselio',
        plugin='ellipselio::MappingNode',
        name='ellipselio',
        parameters=[PathJoinSubstitution([config_path, config_file]),
                    {'use_sim_time': use_sim_time},
                    save_params],
        extra_arguments=[{'use_intra_process_comms': True}],
    )

    composable_node = LoadComposableNodes(
        target_container=container_name,
        composable_node_descriptions=[
            ellipse_lio_comp,
        ],
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
    ld.add_action(OpaqueFunction(function=resolve_save_path))
    ld.add_action(rviz_node)
    ld.add_action(composable_node)
    # ld.add_action(ellipse_lio_node)

    return ld
