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
    # Protocol constants
    PACKET_SIZE = 132
    PACKET_HEADER_1 = 0xCB
    PACKET_HEADER_2 = 0xCF
    CRC_DATA_SIZE = 120
    
    # Data field offsets and sizes
    TENSILE_DATA_OFFSET = 0
    TENSILE_DATA_SIZE = 76  # 19 int32 * 4 bytes
    ACC_DATA_OFFSET = 76
    ACC_DATA_SIZE = 12      # 3 float * 4 bytes
    GYRO_DATA_OFFSET = 88
    GYRO_DATA_SIZE = 12     # 3 float * 4 bytes
    MAG_DATA_OFFSET = 100
    MAG_DATA_SIZE = 12      # 3 float * 4 bytes
    TEMP_DATA_OFFSET = 112
    TEMP_DATA_SIZE = 4      # 1 float * 4 bytes
    TIMESTAMP_OFFSET = 116
    TIMESTAMP_SIZE = 4      # 1 uint32 * 4 bytes
    
    # Sensor constants
    NUM_TENSILE_SENSORS = 19
    NUM_IMU_AXES = 3
    SENSOR_MAX_VALUE = 8192 * 2
    
    # Buffer and timing constants
    DATA_BUFFER_SIZE = 5000
    DEFAULT_CALIBRATION_SAMPLES = 1000
    DEFAULT_BAUDRATE = 1000000
    
    # Validation thresholds
    MAX_ACCELERATION = 100.0    # g-force
    MAX_ANGULAR_VELOCITY = 2000.0  # degrees/second
    
    # Thread timing constants
    QUEUE_TIMEOUT_SHORT = 0.01  # 10ms
    QUEUE_TIMEOUT_LONG = 0.1    # 100ms
    WAIT_TIME_SHORT = 0.001     # 1ms
    WAIT_TIME_MEDIUM = 0.01     # 10ms
    WAIT_TIME_LONG = 1.0        # 1s

    def __init__(self):
        super().__init__('glove_node')
        
        # Publisher setup
        self.glove_publisher = self.create_publisher(GloveDataMsg, '/glove/data', 10)
        
        # Inference mode setup
        self.inference_mode = self.declare_parameter('inference_mode', False).value
        if self.inference_mode:
            self._setup_inference_model()
        
        # Serial port setup
        self.left_serial_port = self._setup_serial_port('left_port', '/dev/ttyUSB0')
        self.right_serial_port = self._setup_serial_port('right_port', '/dev/ttyUSB1')
        
        # Thread control
        self.running = True
        self.is_calibrated = False
        
        # Data buffers for each hand
        self.left_data_buffer = collections.deque(maxlen=self.DATA_BUFFER_SIZE)
        self.right_data_buffer = collections.deque(maxlen=self.DATA_BUFFER_SIZE)
        
        # Calibration parameters for each hand
        self._init_calibration_parameters()
        
        # Data queues for thread communication
        self.left_data_queue = Queue()
        self.right_data_queue = Queue()
        
        # Thread setup
        self._init_threads()

    def _setup_inference_model(self):
        """Initialize ONNX inference model"""
        self.get_logger().info("Inference mode enabled")
        pkg_share = get_package_share_directory('glove_ros')
        model_path = os.path.join(pkg_share, 'model', '20250417_165613_test.onnx')
        self.get_logger().info(f"Loading model from: {model_path}")
        
        if not os.path.exists(model_path):
            self.get_logger().error(f"Model file not found at: {model_path}")
            self.inference_mode = False
        else:
            self.ort_sess = ort.InferenceSession(model_path)

    def _init_calibration_parameters(self):
        """Initialize calibration parameters for both hands"""
        self.calibration_samples_min_max = self.DEFAULT_CALIBRATION_SAMPLES
        self.calibration_samples_avg = self.DEFAULT_CALIBRATION_SAMPLES
        
        # Left hand calibration data
        self.left_min_val = [self.SENSOR_MAX_VALUE] * self.NUM_TENSILE_SENSORS
        self.left_max_val = [0] * self.NUM_TENSILE_SENSORS
        self.left_avg_val = [0.0] * self.NUM_TENSILE_SENSORS
        
        # Right hand calibration data
        self.right_min_val = [self.SENSOR_MAX_VALUE] * self.NUM_TENSILE_SENSORS
        self.right_max_val = [0] * self.NUM_TENSILE_SENSORS
        self.right_avg_val = [0.0] * self.NUM_TENSILE_SENSORS

    def _init_threads(self):
        """Initialize worker threads"""
        self.left_reader_thread = Thread(
            target=self._read_and_publish_data,
            args=(self.left_serial_port, self.left_data_buffer, 'left'),
            daemon=True
        )
        self.right_reader_thread = Thread(
            target=self._read_and_publish_data,
            args=(self.right_serial_port, self.right_data_buffer, 'right'),
            daemon=True
        )
        self.process_thread = Thread(
            target=self._process_and_publish_data,
            daemon=True
        )

    def _setup_serial_port(self, param_name, default_port):
        """Setup serial port with parameters"""
        port = self.declare_parameter(param_name, default_port).value
        
        # Declare baudrate parameter only once
        if not hasattr(self, '_baudrate_declared'):
            self.baudrate = self.declare_parameter('baudrate', self.DEFAULT_BAUDRATE).value
            self._baudrate_declared = True
        
        self.get_logger().info(f"Setting up {param_name} on {port} with baudrate {self.baudrate}")
        
        try:
            ser = serial.Serial(port, self.baudrate, timeout=1)
            self.get_logger().info(f"Successfully opened {param_name} on {port}")
            return ser
        except Exception as e:
            self.get_logger().error(f"Failed to open {param_name} on {port}: {e}")
            raise

    def _is_valid_data(self, data):
        """Validate data packet integrity"""
        try:
            # Check packet length
            if len(data) != self.PACKET_SIZE:
                self.get_logger().error(f"Invalid packet length: {len(data)} bytes")
                return False
            
            # Validate CRC32 checksum
            received_crc = struct.unpack('<I', data[-4:])[0]
            computed_crc = zlib.crc32(data[:self.CRC_DATA_SIZE]) & 0xFFFFFFFF
            
            if computed_crc != received_crc:
                self.get_logger().error("CRC validation failed:")
                self.get_logger().error(f"  - Computed CRC: {computed_crc:08X}")
                self.get_logger().error(f"  - Received CRC: {received_crc:08X}")
                self.get_logger().error(f"  - Data length: {len(data)}")
                return False
            
            return True
            
        except Exception as e:
            self.get_logger().error(f"Data validation error: {e}")
            return False

    def calibrate(self):
        """Perform calibration sequence for both hands"""
        # Left hand min/max calibration
        self.get_logger().info("Starting left hand min/max calibration...")
        input("Please keep your left hand still. Press Enter to start left hand min/max calibration...")
        self._calibrate_min_max('left')
        
        # Right hand min/max calibration
        self.get_logger().info("Starting right hand min/max calibration...")
        input("Please keep your right hand still. Press Enter to start right hand min/max calibration...")
        self._calibrate_min_max('right')
        
        # Left hand static average calibration
        self.get_logger().info("Starting left hand static average calibration...")
        input("Please keep your left hand still. Press Enter to start left hand static average calibration...")
        self._calibrate_static_average('left')
        
        # Right hand static average calibration
        self.get_logger().info("Starting right hand static average calibration...")
        input("Please keep your right hand still. Press Enter to start right hand static average calibration...")
        self._calibrate_static_average('right')
        
        # Start all threads after calibration
        self.is_calibrated = True
        self.get_logger().info("Calibration complete!")
        self.left_reader_thread.start()
        self.right_reader_thread.start()
        self.process_thread.start()

    def _calibrate_min_max(self, hand_type):
        """Calibrate min/max values for specified hand"""
        self.get_logger().info(
            f"Starting {hand_type} hand min/max calibration, "
            f"collecting {self.calibration_samples_min_max} samples..."
        )
        
        # Select hand-specific parameters
        serial_port, data_buffer, min_val, max_val = self._get_hand_parameters(hand_type)
        
        # Reset calibration parameters
        min_val[:] = [self.SENSOR_MAX_VALUE] * self.NUM_TENSILE_SENSORS
        max_val[:] = [0] * self.NUM_TENSILE_SENSORS
        
        # Clear buffer and wait for stable data
        data_buffer.clear()
        self._clear_serial_buffer(serial_port)
        time.sleep(self.WAIT_TIME_LONG)
        
        self._log_serial_port_status(hand_type, serial_port)
        self._wait_for_first_valid_packet(hand_type, serial_port)
        
        # Collect calibration data
        collected = 0
        last_progress = -1
        
        while collected < self.calibration_samples_min_max:
            if serial_port.in_waiting >= self.PACKET_SIZE:
                packet = serial_port.read(self.PACKET_SIZE)
                if len(packet) == self.PACKET_SIZE and self._is_valid_data(packet):
                    hand_data = self._unpack_data(packet, hand_type)
                    if hand_data:
                        self._update_min_max_values(hand_data, min_val, max_val)
                        collected += 1
                        last_progress = self._update_progress_bar(
                            collected, self.calibration_samples_min_max,
                            f"{hand_type} hand calibrating", last_progress
                        )
                else:
                    self.get_logger().debug(f"Invalid data packet for {hand_type} hand")
            else:
                time.sleep(self.WAIT_TIME_SHORT)
        
        print()
        self._log_calibration_results(hand_type, "min/max", min_val, max_val, collected)

    def _calibrate_static_average(self, hand_type):
        """Calibrate static average values for specified hand"""
        self.get_logger().info(
            f"Starting {hand_type} hand static average calibration, "
            f"collecting {self.calibration_samples_avg} samples..."
        )
        
        # Initialize accumulator
        tensile_sums = [0] * self.NUM_TENSILE_SENSORS
        
        # Select hand-specific parameters
        serial_port, data_buffer, _, _ = self._get_hand_parameters(hand_type)
        data_buffer.clear()
        time.sleep(self.WAIT_TIME_LONG)
        
        # Collect calibration data
        collected = 0
        last_progress = -1
        
        while collected < self.calibration_samples_avg:
            if serial_port.in_waiting >= self.PACKET_SIZE:
                packet = serial_port.read(self.PACKET_SIZE)
                if self._is_valid_data(packet):
                    data = self._unpack_data(packet, hand_type)
                    if data:
                        for i in range(self.NUM_TENSILE_SENSORS):
                            tensile_sums[i] += data['tensile_data'][i]
                        collected += 1
                        last_progress = self._update_progress_bar(
                            collected, self.calibration_samples_avg,
                            f"{hand_type} hand calibration in progress", last_progress
                        )
            time.sleep(self.WAIT_TIME_SHORT)
        
        print()
        self.get_logger().info(f"{hand_type} hand static average calibration completed!")
        
        # Calculate and store averages
        if collected > 0:
            avg_values = self.left_avg_val if hand_type == 'left' else self.right_avg_val
            for i in range(self.NUM_TENSILE_SENSORS):
                avg_values[i] = tensile_sums[i] / collected
        
        self.get_logger().info(f"{hand_type} hand average values: {avg_values}")

    def _get_hand_parameters(self, hand_type):
        """Get hand-specific parameters"""
        if hand_type == 'left':
            return (self.left_serial_port, self.left_data_buffer,
                    self.left_min_val, self.left_max_val)
        else:
            return (self.right_serial_port, self.right_data_buffer,
                    self.right_min_val, self.right_max_val)

    def _clear_serial_buffer(self, serial_port):
        """Clear complete packets from serial buffer"""
        while serial_port.in_waiting >= self.PACKET_SIZE:
            data = serial_port.read(self.PACKET_SIZE)
            self.get_logger().debug("Cleared one complete packet of 132 bytes")
        time.sleep(0.1)

    def _log_serial_port_status(self, hand_type, serial_port):
        """Log serial port configuration"""
        self.get_logger().info(f"{hand_type} hand serial port status:")
        self.get_logger().info(f"  - Port: {serial_port.port}")
        self.get_logger().info(f"  - Baudrate: {serial_port.baudrate}")
        self.get_logger().info(f"  - Bytesize: {serial_port.bytesize}")
        self.get_logger().info(f"  - Parity: {serial_port.parity}")
        self.get_logger().info(f"  - Stopbits: {serial_port.stopbits}")

    def _wait_for_first_valid_packet(self, hand_type, serial_port):
        """Wait for the first valid data packet"""
        self.get_logger().info(f"Waiting for first valid data packet from {hand_type} hand...")
        first_packet = None
        
        while first_packet is None:
            if serial_port.in_waiting >= self.PACKET_SIZE:
                packet = serial_port.read(self.PACKET_SIZE)
                if len(packet) == self.PACKET_SIZE and self._is_valid_data(packet):
                    first_packet = packet
                    self.get_logger().info(f"Received first valid packet from {hand_type} hand")
                else:
                    self.get_logger().debug("Invalid first packet, waiting for next...")
            time.sleep(self.WAIT_TIME_SHORT)

    def _update_min_max_values(self, hand_data, min_val, max_val):
        """Update min/max calibration values"""
        for i in range(self.NUM_TENSILE_SENSORS):
            val = hand_data['tensile_data'][i]
            if val < min_val[i]:
                min_val[i] = val
            if val > max_val[i]:
                max_val[i] = val

    def _update_progress_bar(self, current, total, message, last_progress):
        """Update and display progress bar"""
        progress = int((current / total) * 50)
        if progress != last_progress:
            bar = '[' + '#' * progress + '-' * (50 - progress) + ']'
            print(f"\r{message} {bar} {current}/{total}", end='')
            import sys
            sys.stdout.flush()
        return progress

    def _log_calibration_results(self, hand_type, calibration_type, min_val, max_val, collected):
        """Log calibration results"""
        self.get_logger().info(f"{hand_type} hand {calibration_type} calibration completed!")
        self.get_logger().info(f"Recorded min values: {min_val}")
        self.get_logger().info(f"Recorded max values: {max_val}")
        self.get_logger().info(f"Total samples collected: {collected}")

    def _read_and_publish_data(self, serial_port, data_buffer, hand_type):
        """Read serial data and queue for processing"""
        while self.running:
            if serial_port.in_waiting > 0:
                new_data = serial_port.read(serial_port.in_waiting)
                data_buffer.extend(new_data)

                while len(data_buffer) >= self.PACKET_SIZE:
                    packet = bytes([data_buffer.popleft() for _ in range(self.PACKET_SIZE)])
                    if self._is_valid_data(packet):
                        hand_data = self._unpack_data(packet, hand_type)
                        if hand_data:
                            # Queue data for processing
                            if hand_type == 'left':
                                self.left_data_queue.put(hand_data)
                            else:
                                self.right_data_queue.put(hand_data)

    def _process_and_publish_data(self):
        """Process queued data and publish combined messages"""
        left_data = None
        right_data = None
        
        while self.running:
            try:
                # Try to get data from left queue
                if left_data is None:
                    try:
                        left_data = self.left_data_queue.get(timeout=self.QUEUE_TIMEOUT_SHORT)
                    except Empty:
                        pass
                
                # Try to get data from right queue
                if right_data is None:
                    try:
                        right_data = self.right_data_queue.get(timeout=self.QUEUE_TIMEOUT_SHORT)
                    except Empty:
                        pass
                
                # Process and publish when both hands have data
                if left_data is not None and right_data is not None:
                    if self.inference_mode:
                        left_data = self._perform_inference(left_data, 'left')
                        right_data = self._perform_inference(right_data, 'right')
                    
                    self._publish_combined_data(left_data, right_data)
                    
                    # Reset data
                    left_data = None
                    right_data = None
                
                # Wait strategy based on data availability
                elif left_data is not None or right_data is not None:
                    time.sleep(self.WAIT_TIME_SHORT)  # Wait for other hand
                else:
                    time.sleep(self.WAIT_TIME_MEDIUM)  # No data available
                
            except Exception as e:
                self.get_logger().error(f"Error processing data: {e}")
                time.sleep(self.WAIT_TIME_MEDIUM)

    def _perform_inference(self, hand_data, hand_type):
        """Perform inference on hand data"""
        current_tensile = np.array(hand_data['tensile_data']).astype(np.float32)
        avg_calibration = np.array(
            self.left_avg_val if hand_type == 'left' else self.right_avg_val
        ).astype(np.float32)

        # Calculate difference from calibrated baseline
        if current_tensile.shape == avg_calibration.shape:
            tensile_difference = current_tensile - avg_calibration
        else:
            self.get_logger().error(
                f"Shape mismatch for tensile data subtraction: "
                f"current shape: {current_tensile.shape}, "
                f"calibration shape: {avg_calibration.shape}. "
                f"Using raw tensile data for inference."
            )
            tensile_difference = current_tensile

        # Reshape for model input
        model_input = tensile_difference.reshape(1, -1)
        
        # Run inference
        outputs = self.ort_sess.run(None, {'input': model_input})
        hand_data['inference'] = outputs[0][0]
        
        return hand_data

    def _publish_combined_data(self, left_data, right_data):
        """Publish combined left and right hand data"""
        msg = GloveDataMsg()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        
        # Left hand data
        msg.left_linear_acceleration = list(left_data['acc_data'])
        msg.left_angular_velocity = list(left_data['gyro_data'])
        msg.left_temperature = float(left_data['temperature'])
        msg.left_tensile_data = list(left_data['tensile_data'])
        if self.inference_mode:
            msg.left_joint_angles = left_data['inference'].tolist()
        
        # Right hand data
        msg.right_linear_acceleration = list(right_data['acc_data'])
        msg.right_angular_velocity = list(right_data['gyro_data'])
        msg.right_temperature = float(right_data['temperature'])
        msg.right_tensile_data = list(right_data['tensile_data'])
        if self.inference_mode:
            msg.right_joint_angles = right_data['inference'].tolist()
        
        # Use left hand timestamp as reference
        msg.timestamp = int(left_data['timestamp'])
        
        self.glove_publisher.publish(msg)

    def stop(self):
        """Stop all threads and close serial ports"""
        self.running = False
        self.left_reader_thread.join()
        self.right_reader_thread.join()
        self.process_thread.join()
        self.left_serial_port.close()
        self.right_serial_port.close()

    def _unpack_data(self, data, hand_type):
        """Unpack serial data into structured format"""
        try:
            # Unpack data fields with proper offsets
            tensile_data = struct.unpack(
                f'<{self.NUM_TENSILE_SENSORS}i',
                data[self.TENSILE_DATA_OFFSET:self.TENSILE_DATA_OFFSET + self.TENSILE_DATA_SIZE]
            )
            acc_data = struct.unpack(
                f'<{self.NUM_IMU_AXES}f',
                data[self.ACC_DATA_OFFSET:self.ACC_DATA_OFFSET + self.ACC_DATA_SIZE]
            )
            gyro_data = struct.unpack(
                f'<{self.NUM_IMU_AXES}f',
                data[self.GYRO_DATA_OFFSET:self.GYRO_DATA_OFFSET + self.GYRO_DATA_SIZE]
            )
            mag_data = struct.unpack(
                f'<{self.NUM_IMU_AXES}f',
                data[self.MAG_DATA_OFFSET:self.MAG_DATA_OFFSET + self.MAG_DATA_SIZE]
            )
            temperature = struct.unpack(
                '<f',
                data[self.TEMP_DATA_OFFSET:self.TEMP_DATA_OFFSET + self.TEMP_DATA_SIZE]
            )[0]
            timestamp = struct.unpack(
                '<I',
                data[self.TIMESTAMP_OFFSET:self.TIMESTAMP_OFFSET + self.TIMESTAMP_SIZE]
            )[0]

            # Update calibration parameters during operation
            min_val, max_val = self._get_calibration_values(hand_type)
            for i in range(self.NUM_TENSILE_SENSORS):
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

    def _get_calibration_values(self, hand_type):
        """Get calibration values for specified hand"""
        if hand_type == 'left':
            return self.left_min_val, self.left_max_val
        else:
            return self.right_min_val, self.right_max_val


def main(args=None):
    """Main entry point"""
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