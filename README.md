# Glove ROS Package

This package provides a ROS 2 node for interfacing with a data glove over a serial connection. It publishes IMU and tensile sensor data to ROS topics.

## Features
- Reads data from a serial port (default: `/dev/ttyUSB0`)
- Publishes IMU data to `/glove/imu_data` (sensor_msgs/Imu)
- Publishes tensile sensor data to `/glove/tensile_data` (std_msgs/Float32MultiArray)
- Serial port can be set as a ROS parameter

## Usage

### Running the Node
You can run the node and specify parameters such as the serial port and inference mode:

```bash
ros2 run glove_ros serial_node --ros-args -p port:=/dev/ttyUSB0 -p inference_mode:=true
```
If you do not specify the `port` parameter, it defaults to `/dev/ttyUSB0`. The `inference_mode` parameter defaults to `false`.

### Example Launch (Optional)
To set the port in a launch file, add the parameter like this:
```python
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='glove_ros',
            executable='serial_node',
            name='serial_node',
            parameters=[{'port': '/dev/ttyUSB0'}]
        )
    ])
```

## Parameters
- `port` (string): Serial port device (default: `/dev/ttyUSB0`)
- `baudrate` (int): Serial baudrate (default: `1000000`)
- `inference_mode` (bool): Whether to enable inference mode (default: `false`)

## Topics
- `/glove/imu_data` (`sensor_msgs/Imu`): IMU data
- `/glove/tensile_data` (`std_msgs/Float32MultiArray`): Tensile sensor data

## Requirements
- ROS 2 (Foxy, Galactic, Humble, or newer)
- Python packages: `pyserial`, `rclpy`, `sensor_msgs`, `std_msgs`

## Installation
Clone this repository into your ROS 2 workspace `src` folder and build:

```bash
cd ~/ros2_ws/src
# git clone <repo-url> glove_ros
cd ~/ros2_ws
colcon build
```

## License
MIT
