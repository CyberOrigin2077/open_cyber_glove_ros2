#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from glove_ros.msg import GloveDataMsg
import serial
import struct
import zlib
from threading import Thread
import collections
import onnxruntime as ort
import os
import time
import numpy as np
from ament_index_python.packages import get_package_share_directory

class GloveNode(Node):
    def __init__(self):
        super().__init__('glove_node')
        
        self.glove_publisher = self.create_publisher(GloveDataMsg, '/glove/data', 10)
        
        # Declare inference_mode parameter (default: False)
        self.inference_mode = self.declare_parameter('inference_mode', False).value
        if self.inference_mode:
            self.get_logger().info("Inference mode enabled")
            pkg_share = get_package_share_directory('glove_ros')
            model_path = os.path.join(pkg_share, 'model', '20250417_165613_test.onnx')
            self.get_logger().info(f"Loading model from: {model_path}")
            if not os.path.exists(model_path):
                self.get_logger().error(f"Model file not found at: {model_path}")
                self.inference_mode = False
            else:
                self.ort_sess = ort.InferenceSession(model_path)
        
        self.serial_port = self.setup_serial_port()
        self.running = True
        self.is_calibrated = False
        self.data_buffer = collections.deque(maxlen=5000)  # Add a buffer for serial data
        self.reader_thread = Thread(target=self.read_and_publish_data, daemon=True)
        
        self.sensor_max_value = 8192 * 2
        self.min_val = [self.sensor_max_value] * 19
        self.max_val = [0] * 19
        # self.calibration_result = []
        self.avg_val = [0.0] * 19  # 存储19个传感器的平均校准值
        # 校准参数 (可以考虑未来将它们也设为ROS参数)
        self.calibration_samples_min_max = 1000  # 原来的校准样本数
        self.calibration_samples_avg = 1000      # 静置平均校准的样本数

    def setup_serial_port(self):
        port = self.declare_parameter('port', '/dev/ttyUSB0').value
        baudrate = self.declare_parameter('baudrate', 1000000).value
        return serial.Serial(port, baudrate, timeout=1)

    def check_frame_validity(self, data, offset):
        """Check the validity of the data frame
        1. Check if timestamp is within a reasonable range
        2. Check if tensile sensor data is within a reasonable range
        3. Check if IMU data is within a reasonable range
        """
        try:
            # Check tensile sensor data (first 19 int32)
            for i in range(19):
                val = struct.unpack('<i', data[offset+i*4:offset+(i+1)*4])[0]
                # print(f"Tensile sensor {i}: {val}")
                if not (0 <= val <= self.sensor_max_value - 1):  # Tensile value should be in reasonable range
                    print(f"Tensile sensor {i}: {val}")
                    return False
            
            # Check accelerometer data (3 floats)
            acc_offset = offset + 19*4
            for i in range(3):
                val = struct.unpack('<f', data[acc_offset+i*4:acc_offset+(i+1)*4])[0]
                if abs(val) > 100:  # Acceleration generally won't exceed 16g
                    print(f"Accelerometer {i}: {val}")
                    return False
            
            # Check gyroscope data (3 floats)
            gyro_offset = acc_offset + 3*4
            for i in range(3):
                val = struct.unpack('<f', data[gyro_offset+i*4:gyro_offset+(i+1)*4])[0]
                if abs(val) > 2000:  # Angular velocity generally won't exceed 2000 degrees/second
                    print(f"Gyroscope {i}: {val}")
                    return False
            
            return True
        except Exception:
            return False


    def is_valid_data(self, data):
        """Check the validity of the data."""
        try:        
            # Perform CRC32 validation
            received_crc = struct.unpack('<I', data[-4:])[0]
            computed_crc = zlib.crc32(data[:120]) & 0xFFFFFFFF  # Only compute CRC for first 120 bytes
            
            if computed_crc != received_crc:
                self.get_logger().error(f"CRC validation failed: Computed={computed_crc:08X}, Received={received_crc:08X}")
                return False
                
            return True
        except Exception as e:
            self.get_logger().error(f"Data validity check failed: {e}")
            return False

    def calibrate(self):
        import sys
        input("Press Enter to start calibration...")
        self.get_logger().info(f"Starting calibration, collecting {self.calibration_samples_min_max} samples...")
        self.min_val = [self.sensor_max_value] * 19
        self.max_val = [0] * 19
        collected_min_max = 0
        last_progress_min_max = -1
        while collected_min_max < self.calibration_samples_min_max:
            if self.serial_port.in_waiting > 0:
                new_data = self.serial_port.read(self.serial_port.in_waiting)
                self.data_buffer.extend(new_data)
                while len(self.data_buffer) >= 132 and collected_min_max < self.calibration_samples_min_max:
                    packet = bytes([self.data_buffer.popleft() for _ in range(132)])
                    if self.is_valid_data(packet):
                        hand_data = self.unpack_data(packet) # 注意: unpack_data 仍会动态更新 min_val/max_val
                        if hand_data:
                            # 在此阶段也明确更新min_val和max_val
                            for i in range(19):
                                val = hand_data['tensile_data'][i]
                                if val < self.min_val[i]:
                                    self.min_val[i] = val
                                if val > self.max_val[i]:
                                    self.max_val[i] = val
                            collected_min_max += 1
                            progress = int((collected_min_max / self.calibration_samples_min_max) * 50)
                            if progress != last_progress_min_max:
                                bar = '[' + '#' * progress + '-' * (50 - progress) + ']'
                                print(f"\rrCalibrating {bar} {collected_min_max}/{self.calibration_samples_min_max}", end='')
                                sys.stdout.flush()
                                last_progress_min_max = progress
        print()  # Newline after progress bar
        self.get_logger().info(f"Min/max calibration completed!")
        self.get_logger().info(f"Recorded min values: {self.min_val}")
        self.get_logger().info(f"Recorded max values: {self.max_val}")

        # --- 阶段 2: 静置平均值校准 (用户需保持手套静止) ---
        input("Please keep the glove still. Press Enter to start the static average calibration...")
        self.get_logger().info(f"Starting static average calibration, collecting {self.calibration_samples_avg} samples...")
        tensile_sums = [0] * 19  # 用于累加每个传感器的值
        collected_avg = 0
        last_progress_avg = -1
        
        self.data_buffer.clear() 
        while collected_avg < self.calibration_samples_avg:
            if self.serial_port.in_waiting > 0:
                new_data = self.serial_port.read(self.serial_port.in_waiting)
                self.data_buffer.extend(new_data)
                while len(self.data_buffer) >= 132 and collected_avg < self.calibration_samples_avg:
                    packet = bytes([self.data_buffer.popleft() for _ in range(132)])
                    if self.is_valid_data(packet):
                        hand_data = self.unpack_data(packet) # unpack_data 仍会动态更新 min_val/max_val
                        if hand_data:
                            for i in range(19):
                                tensile_sums[i] += hand_data['tensile_data'][i]
                            collected_avg += 1
                            progress = int((collected_avg / self.calibration_samples_avg) * 50)
                            if progress != last_progress_avg:
                                bar = '[' + '#' * progress + '-' * (50 - progress) + ']'
                                print(f"\rStatic average calibration in progress {bar} {collected_avg}/{self.calibration_samples_avg}", end='')
                                sys.stdout.flush()
                                last_progress_avg = progress
        print()
        if collected_avg > 0:
            for i in range(19):
                self.avg_val[i] = tensile_sums[i] / collected_avg
        
        self.get_logger().info(f"Static average calibration completed!")
        self.get_logger().info(f"Calculated average values: {self.avg_val}")

        # 所有校准阶段完成后
        self.is_calibrated = True
        self.get_logger().info(f"Calibration complete!")
        self.reader_thread.start()

    def inference(self, hand_data):
        # tensile_data = np.array(hand_data['tensile_data']).astype(np.float32)
        # outputs = self.ort_sess.run(None, {'input': tensile_data.reshape(1, -1)})
        # hand_data['inference'] = outputs[0][0]
        # return hand_data

        current_tensile_list = hand_data['tensile_data']
        current_tensile_np = np.array(current_tensile_list).astype(np.float32)
        avg_calibration_np = np.array(self.avg_val).astype(np.float32)

        # 计算差值： 当前读数 - 静置时的平均读数
        if current_tensile_np.shape == avg_calibration_np.shape:
            tensile_difference_np = current_tensile_np - avg_calibration_np
            # 你可以在这里添加一个调试日志来查看差值（可选）
            # self.get_logger().debug(f"Tensile difference for model input: {tensile_difference_np.tolist()}")
        else:
            # 这种情况理论上不应该发生，因为 self.avg_val 和 tensile_data 都应该是19个元素
            # 但作为一种保护措施，如果形状不匹配，则记录错误并使用原始数据（或进行其他错误处理）
            self.get_logger().error(
                f"Shape mismatch for tensile data subtraction: "
                f"current_tensile_np shape: {current_tensile_np.shape}, "
                f"avg_calibration_np shape: {avg_calibration_np.shape}. "
                f"Using raw tensile data for inference instead."
            )
            tensile_difference_np = current_tensile_np # 回退到使用原始拉伸数据

        # 将计算得到的差值数组调整为模型期望的输入形状 (1个样本, N个特征)
        model_input_np = tensile_difference_np.reshape(1, -1)
        
        # 使用处理后的差值数据 (model_input_np) 执行ONNX模型推断
        outputs = self.ort_sess.run(None, {'input': model_input_np})
        hand_data['inference'] = outputs[0][0] 
        
        return hand_data


    def read_and_publish_data(self):
        while self.running:
            if self.serial_port.in_waiting > 0:
                new_data = self.serial_port.read(self.serial_port.in_waiting)
                self.data_buffer.extend(new_data)  # Add new data to the buffer

                # Process data in the buffer
                while len(self.data_buffer) >= 132:
                    packet = bytes([self.data_buffer.popleft() for _ in range(132)])  # Extract a packet
                    if self.is_valid_data(packet):  # Check validity
                        hand_data = self.unpack_data(packet)
                        if self.inference_mode:
                            hand_data = self.inference(hand_data)
                        if hand_data:
                            self.publish_data(hand_data)

    def unpack_data(self, data):
        try:
            # First validate CRC
            received_crc = struct.unpack('<I', data[-4:])[0]
            computed_crc = zlib.crc32(data[:120]) & 0xFFFFFFFF
            
            if computed_crc != received_crc:
                self.get_logger().error(f"CRC validation failed: Computed={computed_crc:08X}, Received={received_crc:08X}")
                return None

            # Unpack the data fields separately to avoid alignment issues
            tensile_data = struct.unpack('<19i', data[:76])  # 19 integers (76 bytes)
            acc_data = struct.unpack('<3f', data[76:88])     # 3 floats (12 bytes)
            gyro_data = struct.unpack('<3f', data[88:100])   # 3 floats (12 bytes)
            mag_data = struct.unpack('<3f', data[100:112])   # 3 floats (12 bytes)
            temperature = struct.unpack('<f', data[112:116])[0]  # 1 float (4 bytes)
            timestamp = struct.unpack('<I', data[116:120])[0]    # 1 uint32 (4 bytes)
            # Reserve bytes and CRC are in the remaining 12 bytes

            for i in range(19):
                if tensile_data[i] < self.min_val[i]:
                    self.min_val[i] = tensile_data[i]
                if tensile_data[i] > self.max_val[i]:
                    self.max_val[i] = tensile_data[i]

            return {
                'tensile_data': tensile_data,
                'acc_data': acc_data,
                'gyro_data': gyro_data,
                'mag_data': mag_data,
                'temperature': temperature,
                'timestamp': timestamp
            }
        except struct.error as e:
            self.get_logger().error(f"Failed to unpack data: {e}")
            return None

    def publish_data(self, hand_data):
        msg = GloveDataMsg()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.linear_acceleration = list(hand_data['acc_data'])
        msg.angular_velocity = list(hand_data['gyro_data'])
        msg.temperature = float(hand_data['temperature'])
        msg.tensile_data = list(hand_data['tensile_data'])
        msg.timestamp = int(hand_data['timestamp'])
        if self.inference_mode:
            msg.joint_angles = hand_data['inference'].tolist()
        self.glove_publisher.publish(msg)


    def stop(self):
        self.running = False
        self.reader_thread.join()
        self.serial_port.close()

def main(args=None):
    rclpy.init(args=args)
    node = GloveNode()
    node.calibrate()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()