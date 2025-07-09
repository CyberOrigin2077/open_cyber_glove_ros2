# OpenCyberGlove ROS2 Wrapper

This package provides ROS 2 nodes for interfacing with OpenCyberGlove. It includes both a serial data acquisition node and a 3D hand visualization node.

## 1. Prerequisites and Setup

### OpenCyberGlove Setup
Before using this ROS2 package, you need to set up the OpenCyberGlove library. Please refer to the [OpenCyberGlove documentation](https://github.com/CyberOrigin2077/open_cyber_glove) for detailed setup instructions, including:

- Hardware requirements
- Software dependencies
- Environment configuration

### ROS2 Package Installation
Clone this repository into your ROS 2 workspace `src` folder:
```bash
cd ~/ros2_ws/src
git clone https://github.com/CyberOrigin2077/open_cyber_glove_ros2.git open_cyber_glove_ros2
```

Build the package with `colcon`:
```bash
cd ~/ros2_ws
colcon build --packages-select open_cyber_glove_ros2
```

Source the workspace:
```bash
source install/setup.bash
```

## 2. Running the Nodes

### Serial Node (Data Acquisition)
The `data_node` reads data from CyberGlove devices and publishes it to ROS topics.

#### Basic Usage
```bash
ros2 run open_cyber_glove_ros2 data_node
```

#### With Custom Parameters
```bash
ros2 run open_cyber_glove_ros2 data_node --ros-args \
  -p left_port:=/dev/ttyUSB0 \
  -p right_port:=/dev/ttyUSB1 \
  -p model_path:=/path/to/your/model.onnx \
  -p inference_mode:=true
```

#### Parameters
- `left_port` (string): Serial port for the left glove (default: `/dev/ttyUSB0`)
- `right_port` (string): Serial port for the right glove (default: `/dev/ttyUSB1`)
- `model_path` (string): Path to ONNX model for joint angle inference (default: "")
- `inference_mode` (bool): Enable inference mode to publish joint angles (default: `false`)

#### Examples
```bash
# Run with left glove only
ros2 run open_cyber_glove_ros2 data_node --ros-args -p right_port:=none

# Run with inference enabled
ros2 run open_cyber_glove_ros2 data_node --ros-args \
  -p model_path:=/path/to/model.onnx \
  -p inference_mode:=true
```

### Visualizer Node (3D Hand Visualization)
The `visualizer_node` provides real-time 3D visualization of hand movements using the glove data.

#### Basic Usage
```bash
ros2 run open_cyber_glove_ros2 visualizer_node
```

#### With Custom Parameters
```bash
ros2 run open_cyber_glove_ros2 visualizer_node --ros-args \
  -p hand_model_path:=/path/to/hand/model
```

#### Parameters
- `hand_model_path` (string): Path to hand model for visualization (default: "")

#### Running Both Nodes
You can run both nodes simultaneously in separate terminals:

**Terminal 1:**
```bash
ros2 run open_cyber_glove_ros2 data_node --ros-args \
  -p inference_mode:=true \
  -p model_path:=/path/to/model.onnx
```

**Terminal 2:**
```bash
ros2 run open_cyber_glove_ros2 visualizer_node --ros-args \
  -p hand_model_path:=/path/to/hand/model
```

## 3. Published Topics and Message Definitions

### Serial Node Topics

The `data_node` publishes glove data to separate topics for left and right hands:

#### Left Hand Data
- **Topic**: `/glove/left/data`
- **Message Type**: `open_cyber_glove_ros2/GloveDataMsg`
- **Description**: Raw sensor data and optional joint angles for the left hand

#### Right Hand Data
- **Topic**: `/glove/right/data`
- **Message Type**: `open_cyber_glove_ros2/GloveDataMsg`
- **Description**: Raw sensor data and optional joint angles for the right hand

### GloveDataMsg Message Definition

The `open_cyber_glove_ros2/GloveDataMsg` message contains the following fields:

```msg
std_msgs/Header header          # Standard ROS header with timestamp and frame_id
float32[3] linear_acceleration # Linear acceleration from IMU (m/s²)
float32[3] angular_velocity    # Angular velocity from IMU (rad/s)
float32 temperature            # Temperature sensor reading (°C)
int32[19] tensile_data        # Raw tensile sensor readings (19 sensors)
float32[22] joint_angles      # Computed joint angles (only if inference_mode=true)
uint32 timestamp              # Device timestamp
```

### Message Field Details

- **`header`**: Standard ROS header containing timestamp and frame information
- **`linear_acceleration`**: 3-axis linear acceleration from the glove's IMU sensor
- **`angular_velocity`**: 3-axis angular velocity (gyroscope) from the glove's IMU sensor
- **`temperature`**: Temperature reading from the glove's temperature sensor
- **`tensile_data`**: Array of 19 raw tensile sensor readings from the glove's strain sensors
- **`joint_angles`**: Array of 22 joint angles computed by the inference model (only published when `inference_mode=true`)
- **`timestamp`**: Device timestamp for data synchronization

### Visualizer Node Subscriptions

The `visualizer_node` subscribes to the same topics to visualize the hand movements:

- **Subscribes to**: `/glove/left/data` and `/glove/right/data`
- **Message Type**: `open_cyber_glove_ros2/GloveDataMsg`
- **Function**: Updates 3D hand visualization based on joint angles

## License
BSD 3-Clause License

Copyright (c) 2024, CyberOrigin Contributors
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
