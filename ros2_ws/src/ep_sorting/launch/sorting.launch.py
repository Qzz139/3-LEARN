"""EP camera + YOLO + sorting action server + optional one-shot task client."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from ament_index_python.packages import get_package_share_directory
from pathlib import Path


def generate_launch_description():
    share = Path(get_package_share_directory('ep_sorting'))
    return LaunchDescription([
        DeclareLaunchArgument('execute', default_value='false'),
        DeclareLaunchArgument('model_path', default_value=str(share / 'models/yolo26m.pt')),
        DeclareLaunchArgument('config_path', default_value=str(share / 'config/object_sorting.json')),
        DeclareLaunchArgument('state_machine_path',
                              default_value=str(share / 'config/sorting_state_machine.yaml')),
        DeclareLaunchArgument('log_root', default_value=str(Path.home() / 'projects/3-LEARN/work')),
        Node(package='ep_sorting', executable='ep_hardware', output='screen', parameters=[{
            'config_path': LaunchConfiguration('config_path'),
            'state_machine_path': LaunchConfiguration('state_machine_path'),
            'log_root': LaunchConfiguration('log_root'),
        }]),
        Node(package='ep_sorting', executable='ep_detector', output='screen', parameters=[{
            'model_path': LaunchConfiguration('model_path'),
        }]),
        Node(package='ep_sorting', executable='ep_task', output='screen', parameters=[{
            'execute': LaunchConfiguration('execute'),
        }]),
    ])
