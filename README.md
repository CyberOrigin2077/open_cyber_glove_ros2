# CyberGlove ROS2 Driver

This package provides a ROS 2 node for interfacing with CyberGlove systems over a serial connection. It publishes processed glove data, including sensor readings and IMU data, to ROS topics for both left and right hands.

## 1. Setup

### Installation
First, clone this repository into your ROS 2 workspace `src` folder:
```bash
cd ~/ros2_ws/src
# git clone <repo-url> glove_ros
```

### Dependencies
Install the required Python packages using `pip`:
```bash
pip install -r requirements.txt
```

Then, build the package with `colcon`:
```bash
cd ~/ros2_ws
colcon build --packages-select glove_ros
```

## 2. Usage

### Running the Node
You can run the node using `ros2 run`. You can specify parameters for serial ports and other settings.

```bash
ros2 run glove_ros serial_node --ros-args -p left_port:=/dev/ttyUSB0 -p right_port:=/dev/ttyUSB1 -p inference_mode:=true
```

### Parameters
- `left_port` (string): Serial port for the left glove (default: `/dev/ttyUSB0`).
- `right_port` (string): Serial port for the right glove (default: `/dev/ttyUSB1`).
- `baudrate` (int): Serial baudrate (default: `1000000`).
- `inference_mode` (bool): Enable inference mode to publish joint angles (default: `false`).

## 3. Published Topics

This node publishes data for the left and right gloves to separate topics.

- **Topic**: `/glove/left/data`
- **Message Type**: `glove_ros/GloveDataMsg`

- **Topic**: `/glove/right/data`
- **Message Type**: `glove_ros/GloveDataMsg`

### `GloveDataMsg` Message Format
The `glove_ros/GloveDataMsg` contains the following fields:
```
std_msgs/Header header
float32[3] linear_acceleration
float32[3] angular_velocity
float32 temperature
int32[19] tensile_data
float32[22] joint_angles  # Published only if inference_mode is true
uint32 timestamp
```

## License
BSD 3-Clause License

Copyright (c) 2024, OpenCyberGlove Contributors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
