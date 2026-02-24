from setuptools import find_packages, setup
import os
from glob import glob
from setuptools import setup

package_name = 'a200_navigation'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # This installs everything in your launch folder to the share directory
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*.[pxy][yam]*'))),
        # This installs your robot.yaml from the resource folder
        (os.path.join('share', package_name, 'resource'), glob(os.path.join('resource', '*.[pxy][yam]*'))),
    ],
    
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='tgunther',
    maintainer_email='tgunther@ufl.edu',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'hello_world = a200_navigation.hello_world:main'
        ],
    },
)
