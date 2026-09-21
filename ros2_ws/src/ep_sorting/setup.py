from glob import glob
from os.path import relpath
from pathlib import Path
from setuptools import setup


package_name = 'ep_sorting'
root = Path(__file__).resolve().parents[3]
package_dir = Path(__file__).resolve().parent
share = 'share/' + package_name


def source(path):
    # colcon requires setup.py data_files sources relative to this package.
    return relpath(path, package_dir)

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        (share, ['package.xml']),
        (share + '/launch', glob('launch/*.launch.py')),
        (share + '/legacy', [source(path) for path in sorted((root / 'scripts').glob('*.py'))]),
        (share + '/config', [source(root / 'config/object_sorting.json'),
                            source(root / 'config/sorting_state_machine.yaml')]),
        (share + '/models', [source(root / 'models/yolo26m.pt'),
                            source(root / 'models/yolo26m.source.json')]),
    ],
    install_requires=['setuptools'],
    zip_safe=False,
    maintainer='EP sorting team',
    maintainer_email='robot@example.invalid',
    description='ROS 2 integration for RoboMaster EP six-object sorting',
    url='https://github.com/Qzz139/3-LEARN',
    license='MIT',
    entry_points={'console_scripts': [
        'ep_hardware = ep_sorting.hardware:main',
        'ep_detector = ep_sorting.detector:main',
        'ep_task = ep_sorting.task:main',
    ]},
)
