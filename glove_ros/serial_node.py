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
from queue import Queue
from queue import Empty

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
        
        # 设置两个串口
        self.left_serial_port = self.setup_serial_port('left_port', '/dev/ttyUSB0')
        self.right_serial_port = self.setup_serial_port('right_port', '/dev/ttyUSB1')
        
        self.running = True
        self.is_calibrated = False
        
        # 为每个串口创建独立的数据缓冲区
        self.left_data_buffer = collections.deque(maxlen=5000)
        self.right_data_buffer = collections.deque(maxlen=5000)
        
        # 为左右手分别创建校准参数
        self.sensor_max_value = 8192 * 2
        self.left_min_val = [self.sensor_max_value] * 19
        self.left_max_val = [0] * 19
        self.left_avg_val = [0.0] * 19
        self.right_min_val = [self.sensor_max_value] * 19
        self.right_max_val = [0] * 19
        self.right_avg_val = [0.0] * 19
        
        # 校准参数
        self.calibration_samples_min_max = 1000
        self.calibration_samples_avg = 1000
        
        # 创建数据队列用于线程间通信
        self.data_queue = Queue()
        
        # 创建两个独立的读取线程和一个处理线程
        self.left_reader_thread = Thread(target=self.read_and_publish_data, args=(self.left_serial_port, self.left_data_buffer, 'left'), daemon=True)
        self.right_reader_thread = Thread(target=self.read_and_publish_data, args=(self.right_serial_port, self.right_data_buffer, 'right'), daemon=True)
        self.process_thread = Thread(target=self.process_and_publish_data, daemon=True)

    def setup_serial_port(self, param_name, default_port):
        """设置串口参数"""
        port = self.declare_parameter(param_name, default_port).value
        # 只在第一次调用时声明baudrate参数
        if not hasattr(self, '_baudrate_declared'):
            self.baudrate = self.declare_parameter('baudrate', 1000000).value
            self._baudrate_declared = True
        return serial.Serial(port, self.baudrate, timeout=1)

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
        """检查数据包的有效性"""
        try:
            if len(data) != 132:
                return False
                
            # 检查CRC32
            received_crc = struct.unpack('<I', data[-4:])[0]
            computed_crc = zlib.crc32(data[:120]) & 0xFFFFFFFF
            
            if computed_crc != received_crc:
                self.get_logger().error(f"CRC validation failed: Computed={computed_crc:08X}, Received={received_crc:08X}")
                return False
            
            # 检查拉伸传感器数据范围
            for i in range(19):
                val = struct.unpack('<i', data[i*4:(i+1)*4])[0]
                if not (0 <= val <= self.sensor_max_value - 1):
                    return False
            
            # 检查IMU数据范围
            acc_offset = 76
            gyro_offset = 88
            for i in range(3):
                acc_val = struct.unpack('<f', data[acc_offset+i*4:acc_offset+(i+1)*4])[0]
                gyro_val = struct.unpack('<f', data[gyro_offset+i*4:gyro_offset+(i+1)*4])[0]
                if abs(acc_val) > 100 or abs(gyro_val) > 2000:
                    return False
            
            return True
        except Exception as e:
            self.get_logger().error(f"Data validation error: {e}")
            return False

    def calibrate(self):
        import sys
        # 左手最大最小值校准
        self.get_logger().info("Starting left hand min/max calibration...")
        input("Press Enter to start left hand min/max calibration...")
        self.calibrate_min_max('left')
        
        # 右手最大最小值校准
        self.get_logger().info("Starting right hand min/max calibration...")
        input("Press Enter to start right hand min/max calibration...")
        self.calibrate_min_max('right')
        
        # 同时进行左右手静止平均值校准
        self.get_logger().info("Starting static average calibration for both hands...")
        input("Please keep both hands still. Press Enter to start the static average calibration...")
        self.calibrate_static_average()
        
        # 所有校准阶段完成后
        self.is_calibrated = True
        self.get_logger().info("Calibration complete!")
        self.left_reader_thread.start()
        self.right_reader_thread.start()

    def calibrate_min_max(self, hand_type):
        """为指定的手进行最大最小值校准"""
        import sys
        self.get_logger().info(f"Starting {hand_type} hand min/max calibration, collecting {self.calibration_samples_min_max} samples...")
        
        # 选择对应的串口、缓冲区和校准参数
        serial_port = self.left_serial_port if hand_type == 'left' else self.right_serial_port
        data_buffer = self.left_data_buffer if hand_type == 'left' else self.right_data_buffer
        min_val = self.left_min_val if hand_type == 'left' else self.right_min_val
        max_val = self.left_max_val if hand_type == 'left' else self.right_max_val
        
        # 重置校准参数
        min_val[:] = [self.sensor_max_value] * 19
        max_val[:] = [0] * 19
        
        collected_min_max = 0
        last_progress_min_max = -1
        
        # 清空缓冲区
        data_buffer.clear()
        
        # 等待数据稳定
        time.sleep(1)
        
        self.get_logger().info(f"Starting data collection for {hand_type} hand...")
        
        while collected_min_max < self.calibration_samples_min_max:
            if serial_port.in_waiting > 0:
                new_data = serial_port.read(serial_port.in_waiting)
                data_buffer.extend(new_data)
                self.get_logger().debug(f"Read {len(new_data)} bytes for {hand_type} hand")
                
                while len(data_buffer) >= 132 and collected_min_max < self.calibration_samples_min_max:
                    packet = bytes([data_buffer.popleft() for _ in range(132)])
                    if self.is_valid_data(packet):
                        hand_data = self.unpack_data(packet, hand_type)
                        if hand_data:
                            for i in range(19):
                                val = hand_data['tensile_data'][i]
                                if val < min_val[i]:
                                    min_val[i] = val
                                if val > max_val[i]:
                                    max_val[i] = val
                            collected_min_max += 1
                            progress = int((collected_min_max / self.calibration_samples_min_max) * 50)
                            if progress != last_progress_min_max:
                                bar = '[' + '#' * progress + '-' * (50 - progress) + ']'
                                print(f"\r{hand_type} hand calibrating {bar} {collected_min_max}/{self.calibration_samples_min_max}", end='')
                                sys.stdout.flush()
                                last_progress_min_max = progress
                    else:
                        self.get_logger().debug(f"Invalid data packet for {hand_type} hand")
            else:
                # 如果没有数据，等待一小段时间
                time.sleep(0.001)
        
        print()
        self.get_logger().info(f"{hand_type} hand min/max calibration completed!")
        self.get_logger().info(f"Recorded min values: {min_val}")
        self.get_logger().info(f"Recorded max values: {max_val}")
        self.get_logger().info(f"Total samples collected: {collected_min_max}")

    def calibrate_static_average(self):
        """同时进行左右手的静止平均值校准"""
        import sys
        self.get_logger().info(f"Starting static average calibration, collecting {self.calibration_samples_avg} samples...")
        
        # 初始化累加器
        left_tensile_sums = [0] * 19
        right_tensile_sums = [0] * 19
        collected_avg = 0
        last_progress_avg = -1
        
        # 清空缓冲区
        self.left_data_buffer.clear()
        self.right_data_buffer.clear()
        
        # 等待数据稳定
        time.sleep(1)
        
        while collected_avg < self.calibration_samples_avg:
            # 处理左手数据
            if self.left_serial_port.in_waiting >= 132:
                left_packet = self.left_serial_port.read(132)
                if self.is_valid_data(left_packet):
                    left_data = self.unpack_data(left_packet, 'left')
                    if left_data:
                        for i in range(19):
                            left_tensile_sums[i] += left_data['tensile_data'][i]
            
            # 处理右手数据
            if self.right_serial_port.in_waiting >= 132:
                right_packet = self.right_serial_port.read(132)
                if self.is_valid_data(right_packet):
                    right_data = self.unpack_data(right_packet, 'right')
                    if right_data:
                        for i in range(19):
                            right_tensile_sums[i] += right_data['tensile_data'][i]
                        collected_avg += 1
                        progress = int((collected_avg / self.calibration_samples_avg) * 50)
                        if progress != last_progress_avg:
                            bar = '[' + '#' * progress + '-' * (50 - progress) + ']'
                            print(f"\rStatic average calibration in progress {bar} {collected_avg}/{self.calibration_samples_avg}", end='')
                            sys.stdout.flush()
                            last_progress_avg = progress
            
            # 短暂等待，避免CPU占用过高
            time.sleep(0.001)
        
        print()
        # 计算平均值
        if collected_avg > 0:
            for i in range(19):
                self.left_avg_val[i] = left_tensile_sums[i] / collected_avg
                self.right_avg_val[i] = right_tensile_sums[i] / collected_avg
        
        self.get_logger().info("Static average calibration completed!")
        self.get_logger().info(f"Left hand average values: {self.left_avg_val}")
        self.get_logger().info(f"Right hand average values: {self.right_avg_val}")

    def read_and_publish_data(self, serial_port, data_buffer, hand_type):
        """读取串口数据并放入队列"""
        while self.running:
            if serial_port.in_waiting > 0:
                new_data = serial_port.read(serial_port.in_waiting)
                data_buffer.extend(new_data)

                while len(data_buffer) >= 132:
                    packet = bytes([data_buffer.popleft() for _ in range(132)])
                    if self.is_valid_data(packet):
                        hand_data = self.unpack_data(packet, hand_type)
                        if hand_data:
                            # 将数据放入队列，包含手部类型信息
                            self.data_queue.put((hand_type, hand_data))

    def process_and_publish_data(self):
        """处理队列中的数据并发布"""
        left_data = None
        right_data = None
        
        while self.running:
            try:
                # 从队列中获取数据，设置超时以便能够响应停止信号
                hand_type, hand_data = self.data_queue.get(timeout=0.1)
                
                # 根据手部类型存储数据
                if hand_type == 'left':
                    left_data = hand_data
                else:
                    right_data = hand_data
                
                # 当两个手都有数据时，进行推理和发布
                if left_data is not None and right_data is not None:
                    # 如果启用推理模式，分别对左右手数据进行推理
                    if self.inference_mode:
                        left_data = self.inference(left_data, 'left')
                        right_data = self.inference(right_data, 'right')
                    
                    # 发布合并后的数据
                    self.publish_data(left_data, right_data)
                    
                    # 重置数据
                    left_data = None
                    right_data = None
                
            except Empty:
                # 队列为空，继续等待
                continue
            except Exception as e:
                self.get_logger().error(f"Error processing data: {e}")

    def inference(self, hand_data, hand_type):
        """对指定手的数据进行推理"""
        current_tensile_list = hand_data['tensile_data']
        current_tensile_np = np.array(current_tensile_list).astype(np.float32)
        avg_calibration_np = np.array(self.left_avg_val if hand_type == 'left' else self.right_avg_val).astype(np.float32)

        # 计算差值： 当前读数 - 静置时的平均读数
        if current_tensile_np.shape == avg_calibration_np.shape:
            tensile_difference_np = current_tensile_np - avg_calibration_np
        else:
            self.get_logger().error(
                f"Shape mismatch for tensile data subtraction: "
                f"current_tensile_np shape: {current_tensile_np.shape}, "
                f"avg_calibration_np shape: {avg_calibration_np.shape}. "
                f"Using raw tensile data for inference instead."
            )
            tensile_difference_np = current_tensile_np

        # 将计算得到的差值数组调整为模型期望的输入形状 (1个样本, N个特征)
        model_input_np = tensile_difference_np.reshape(1, -1)
        
        # 使用处理后的差值数据执行ONNX模型推断
        outputs = self.ort_sess.run(None, {'input': model_input_np})
        hand_data['inference'] = outputs[0][0] 
        
        return hand_data

    def publish_data(self, left_data, right_data):
        """发布左右手数据"""
        msg = GloveDataMsg()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        
        # 左手数据
        msg.left_linear_acceleration = list(left_data['acc_data'])
        msg.left_angular_velocity = list(left_data['gyro_data'])
        msg.left_temperature = float(left_data['temperature'])
        msg.left_tensile_data = list(left_data['tensile_data'])
        if self.inference_mode:
            msg.left_joint_angles = left_data['inference'].tolist()
        
        # 右手数据
        msg.right_linear_acceleration = list(right_data['acc_data'])
        msg.right_angular_velocity = list(right_data['gyro_data'])
        msg.right_temperature = float(right_data['temperature'])
        msg.right_tensile_data = list(right_data['tensile_data'])
        if self.inference_mode:
            msg.right_joint_angles = right_data['inference'].tolist()
        
        # 使用左手的时间戳（或者可以取两个时间戳的平均值）
        msg.timestamp = int(left_data['timestamp'])
        
        self.glove_publisher.publish(msg)

    def stop(self):
        self.running = False
        self.left_reader_thread.join()
        self.right_reader_thread.join()
        self.process_thread.join()
        self.left_serial_port.close()
        self.right_serial_port.close()

    def unpack_data(self, data, hand_type):
        """解析串口数据"""
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

            # 根据手部类型选择对应的校准参数
            min_val = self.left_min_val if hand_type == 'left' else self.right_min_val
            max_val = self.left_max_val if hand_type == 'left' else self.right_max_val

            for i in range(19):
                if tensile_data[i] < min_val[i]:
                    min_val[i] = tensile_data[i]
                if tensile_data[i] > max_val[i]:
                    max_val[i] = tensile_data[i]

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