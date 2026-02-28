from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'robot_arm'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        # ROS index
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),

        # package.xml
        ('share/' + package_name, ['package.xml']),

        # launch files
        ('share/' + package_name + '/launch', glob('launch/*.py')),

        # urdf / xacro
        ('share/' + package_name + '/urdf', glob('urdf/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='isaac',
    maintainer_email='isaac@todo.todo',
    description='Robot arm description and launch files',
    license='TODO',
    entry_points={
        'console_scripts': [
            'IK = robot_arm.testarm:main',
        ],
    },
)
