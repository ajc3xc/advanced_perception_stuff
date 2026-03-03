from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'fairino10_v6_gazebo'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        # Required: register the package with ROS
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),

        # Install our directories so ros2 launch can find them
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.py')),
        (os.path.join('share', package_name, 'worlds'),
            glob('worlds/*')),
        (os.path.join('share', package_name, 'config'),
            glob('config/*')),
        (os.path.join('share', package_name, 'urdf'),
            glob('urdf/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your name',
    maintainer_email='you@example.com',
    description='Gazebo Harmonic simulation for Fairino FR10 V6',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'planning_node = fairino10_v6_gazebo.path_and_avoid:main',
        ],
    },
)
