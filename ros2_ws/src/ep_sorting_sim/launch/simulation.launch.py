"""One command starts Gazebo, the existing YOLO detector, and the sim sorter."""
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    sim_share = Path(get_package_share_directory('ep_sorting_sim'))
    real_share = Path(get_package_share_directory('ep_sorting'))
    gazebo_share = Path(get_package_share_directory('gazebo_ros'))
    return LaunchDescription([
        DeclareLaunchArgument('execute', default_value='false'),
        DeclareLaunchArgument('model_path', default_value=str(real_share / 'models/yolo26m.pt')),
        DeclareLaunchArgument('log_root', default_value=str(Path.home() / 'projects/3-LEARN/work')),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(gazebo_share / 'launch/gazebo.launch.py')),
            launch_arguments={'world': str(sim_share / 'worlds/sorting.world')}.items()),
        Node(package='ep_sorting', executable='ep_detector', output='screen', parameters=[{
            'model_path': LaunchConfiguration('model_path'),
            'confidence': 0.04,
            'image_size': 960,
        }]),
        Node(package='ep_sorting_sim', executable='sim_sorter', output='screen', parameters=[{
            'execute': LaunchConfiguration('execute'),
            'log_root': LaunchConfiguration('log_root'),
        }]),
    ])
