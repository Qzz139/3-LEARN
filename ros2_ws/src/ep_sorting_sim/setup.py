from glob import glob
from setuptools import setup


package_name = 'ep_sorting_sim'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'README.md']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/worlds', glob('worlds/*.world')),
        ('share/' + package_name + '/meshes', glob('meshes/*.obj')),
    ],
    install_requires=['setuptools'],
    zip_safe=False,
    maintainer='EP sorting team',
    maintainer_email='robot@example.invalid',
    description='Gazebo Classic simulation for six-object sorting',
    license='MIT',
    entry_points={'console_scripts': [
        'sim_sorter = ep_sorting_sim.sorter:main',
    ]},
)
