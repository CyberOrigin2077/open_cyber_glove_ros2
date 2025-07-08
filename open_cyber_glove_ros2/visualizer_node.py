#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

# ROS 2 标准和自定义消息类型
from open_cyber_glove_ros2.msg import GloveDataMsg
from open_cyber_glove.visualizer import HandVisualizer


class VisualizerNode(Node):
    def __init__(self):
        super().__init__('visualizer_node')

        self.hand_model_path = self.declare_parameter("hand_model_path", "").value
        if self.hand_model_path == "":
            self.get_logger().warn("Hand model path is not specified, set inference_mode to False")
            self.hand_model_path = None

        self.visualizer = HandVisualizer(model_path=self.hand_model_path)
        self._create_ros_interfaces()

    def _create_ros_interfaces(self):
        qos_profile = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=10)
        
        self.create_subscription(GloveDataMsg, '/glove/left/data', self.left_callback, qos_profile)
        self.get_logger().info("Subscribed to LEFT hand topic: /glove/left/data")
    
        self.create_subscription(GloveDataMsg, '/glove/right/data', self.right_callback, qos_profile)
        self.get_logger().info("Subscribed to RIGHT hand topic: /glove/right/data")

        # if self.publish_rviz_markers_flag:
        #     self.marker_publisher = self.create_publisher(MarkerArray, '/hand_kinematics_markers', 10)
        #     self.get_logger().info(f"Publishing RViz markers to {self.marker_publisher.topic_name}")

    def left_callback(self, msg: GloveDataMsg):
        self.base_callback(msg, "left")

    def right_callback(self, msg: GloveDataMsg):
        self.base_callback(msg, "right")

    def base_callback(self, msg: GloveDataMsg, hand_type: str):
        self.visualizer.update(msg.joint_angles, hand_type)

    def destroy_node(self):
        self.visualizer.stop()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = VisualizerNode()
        # 既然是硬编码，我们直接检查模型是否初始化成功
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard interrupt, shutting down.')
    finally:
        if node:
            node.destroy_node()
        if rclpy.ok():
            rclpy.try_shutdown()

if __name__ == '__main__':
    main() 