'''
cd ~/one-leg-robot/ros2_ws
colcon build
source ~/one-leg-robot/ros2_ws/install/setup.bash
ros2 launch simulation view_leg.py
'''

from launch import LaunchDescription
from launch_ros.actions import Node
import os

def generate_launch_description():

    # build the full path to the URDF file
    urdf_path = os.path.join(
        os.getenv("HOME"),
        "one-leg-robot",
        "ros2_ws",
        "src",
        "simulation",
        "models",
        "one_leg_robot.urdf"
    )

    print("URDF path:", urdf_path) # show path
    print("URDF exists:", os.path.exists(urdf_path)) # confirm that file was found

    return LaunchDescription([
        Node(
            package='rviz2',                # RViz package
            executable='rviz2',             # RViz program
            name='rviz2'                    # node name
        ),
        Node(
            package='joint_state_publisher_gui',     # slider GUI package
            executable='joint_state_publisher_gui',  # slider executable
            name='joint_state_publisher_gui'         # node name
        ),
        Node(
            package='robot_state_publisher',                 # publishes TF from URDF
            executable='robot_state_publisher',              # robot_state_publisher program
            name='robot_state_publisher',                    # node name
            parameters=[{
                'robot_description': open(urdf_path).read()  # load URDF text
            }]
        )
    ])