#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from open_cyber_glove_ros2.msg import GloveDataMsg
from threading import Thread
import time
import numpy as np
from queue import Empty
from open_cyber_glove import OpenCyberGlove

class GloveNode(Node):
    def __init__(self):
        super().__init__('glove_node')
        self.WAIT_TIME_SHORT = 0.01
        self.WAIT_TIME_MEDIUM = 1.0
        
        # Publisher setup - separate publishers for each hand
        self.left_glove_publisher = self.create_publisher(GloveDataMsg, '/glove/left/data', 10)
        self.right_glove_publisher = self.create_publisher(GloveDataMsg, '/glove/right/data', 10)
        
        # Inference mode setup
        self.inference_mode = self.declare_parameter('inference_mode', False).value
        
        # Serial port setup
        self.left_serial_port = self._setup_serial_port('left_port', None)
        self.right_serial_port = self._setup_serial_port('right_port', None)

        # Model path setup
        model_path = self.declare_parameter("model_path", "").value
        if model_path == "":
            self.get_logger().warn("Model path is not specified, set inference_mode to False")
            self.inference_mode = False
        
        self.glove = OpenCyberGlove(left_port=self.left_serial_port, right_port=self.right_serial_port, model_path=model_path)
        self.glove.start()
        self.glove.calibrate()
        self.get_logger().info(f"Calibration complete, starting to publish data...")

        self._init_threads()

    def _setup_serial_port(self, param_name, default_port):
        """Setup serial port with parameters, return None if device not available"""
        port = self.declare_parameter(param_name, default_port).value
        
        # If port is explicitly disabled, return None
        if port is None or port.lower() in ["none", "null", ""]:
            self.get_logger().info(f"{param_name} is disabled or not specified")
            return None
        
        return port

    def _init_threads(self):
        """Initialize worker threads only for available devices"""
        self.left_process_thread = None
        self.right_process_thread = None
        
        # Only create threads for available devices
        if self.left_serial_port is not None:
            self.left_process_thread = Thread(
                target=self._process_hand_data,
                args=('left',),
                daemon=True
            )
            self.left_process_thread.start()
        
        if self.right_serial_port is not None:
            self.right_process_thread = Thread(
                target=self._process_hand_data,
                args=('right',),
                daemon=True
            )
            self.right_process_thread.start()
    
    def _process_hand_data(self, hand_type):
        """Process queued data and publish for specific hand"""
        while rclpy.ok():
            try:
                data = self.glove.get_data(hand_type=hand_type)
                angles = np.zeros(22)
                if self.inference_mode:
                    angles = self.glove.get_angles(hand_type=hand_type, method='model')

                # Format the data for publishing
                hand_data = {
                    'tensile_data': data.tensile_data,
                    'acc_data': data.acc_data,
                    'gyro_data': data.gyro_data,
                    'mag_data': data.mag_data,
                    'temperature': data.temperature,
                    'timestamp': data.timestamp,
                    'inference': angles,
                }
                                
                # Publish the processed data
                self._publish_hand_data(hand_data, hand_type)
                
            except Empty:
                # No data available, continue with small delay
                time.sleep(self.WAIT_TIME_SHORT)
            except Exception as e:
                self.get_logger().error(f"Error processing {hand_type} hand data: {e}")
                time.sleep(self.WAIT_TIME_MEDIUM)


    def _publish_hand_data(self, hand_data, hand_type):
        """Publish hand data for specified hand type"""
        msg = GloveDataMsg()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.linear_acceleration = list(hand_data['acc_data'])
        msg.angular_velocity = list(hand_data['gyro_data'])
        msg.temperature = float(hand_data['temperature'])
        msg.tensile_data = list(hand_data['tensile_data'])
        msg.timestamp = int(hand_data['timestamp'])
        msg.joint_angles = hand_data['inference'].tolist()
        
        publisher = self.left_glove_publisher if hand_type == 'left' else self.right_glove_publisher
        publisher.publish(msg)

    def stop(self):
        """Stop all threads and close serial ports"""        
        # Join threads if they exist
        if self.left_process_thread is not None:
            self.left_process_thread.join()
        if self.right_process_thread is not None:
            self.right_process_thread.join()
        self.glove.stop()


def main(args=None):
    """Main entry point"""
    rclpy.init(args=args)
    
    try:
        node = GloveNode()
        
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        finally:
            node.stop()
            node.destroy_node()
            
    except SystemExit:
        # Handle graceful exit when no devices are available
        pass
    except Exception as e:
        self.get_logger().error(f"Error during initialization: {e}")
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
