#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from open_cyber_glove_ros2.msg import GloveDataMsg
from open_cyber_glove.visualizer import HandVisualizer
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import Header


class VisualizerNode(Node):
    def __init__(self):
        super().__init__('visualizer_node')

        self.hand_model_path = self.declare_parameter("hand_model_path", "").value
        self.publish_joints = self.declare_parameter("publish_joints", False).value
        if self.hand_model_path == "":
            self.get_logger().warn(
                "Hand model path is not specified, set inference_mode to False")
            self.hand_model_path = None

        self.visualizer = HandVisualizer(model_path=self.hand_model_path)
        self._create_ros_interfaces()

    def _create_ros_interfaces(self):
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1)

        self.create_subscription(GloveDataMsg, '/glove/left/data',
                               self.left_callback, qos_profile)
        self.get_logger().info("Subscribed to LEFT hand topic: /glove/left/data")

        self.create_subscription(GloveDataMsg, '/glove/right/data',
                               self.right_callback, qos_profile)
        self.get_logger().info("Subscribed to RIGHT hand topic: /glove/right/data")

        if self.publish_joints:
            self.joints_publisher = self.create_publisher(
                MarkerArray, '/joints_position', 10)
            self.get_logger().info(
                f"Publishing joints position to {self.joints_publisher.topic_name}")

    def publish_joint_markers(self, msg: GloveDataMsg, hand_type: str):
        joints = self.visualizer.get_joints(msg.joint_angles, hand_type)
        joints_marker_array = MarkerArray()
        fixed_frame_id = "world"

        for i, pos_m in enumerate(joints):
            marker = Marker(
                header=Header(stamp=msg.header.stamp, frame_id=fixed_frame_id),
                ns=f"{hand_type}_joints",
                id=i,
                type=Marker.SPHERE,
                action=Marker.ADD)
            marker.pose.position = Point(
                x=float(pos_m[0]),
                y=float(pos_m[1]),
                z=float(pos_m[2]))
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.01
            marker.scale.y = 0.01
            marker.scale.z = 0.01
            marker.color.a = 0.9
            marker.color.r = 1.0 if hand_type == "left" else 0.1
            marker.color.g = 0.5 if hand_type == "left" else 1.0
            marker.color.b = 0.1 if hand_type == "left" else 0.5
            joints_marker_array.markers.append(marker)

        hand_connections = self.visualizer.HAND_CONNECTIONS
        for bone_idx, (idx_start, idx_end) in enumerate(hand_connections):
            if not (0 <= idx_start < len(joints)
                   and 0 <= idx_end < len(joints)):
                continue
            marker = Marker(
                header=Header(stamp=msg.header.stamp, frame_id=fixed_frame_id),
                ns=f"{hand_type}_links",
                id=len(joints) + bone_idx,
                type=Marker.LINE_STRIP,
                action=Marker.ADD)
            marker.scale.x = 0.005
            marker.color.a = 0.9
            marker.color.r = 0.5 if hand_type == "left" else 0.8
            marker.color.g = 0.5 if hand_type == "left" else 0.2
            marker.color.b = 1.0 if hand_type == "left" else 0.8
            p_start = Point(
                x=float(joints[idx_start][0]),
                y=float(joints[idx_start][1]),
                z=float(joints[idx_start][2]))
            p_end = Point(
                x=float(joints[idx_end][0]),
                y=float(joints[idx_end][1]),
                z=float(joints[idx_end][2]))
            marker.points = [p_start, p_end]
            joints_marker_array.markers.append(marker)

        self.joints_publisher.publish(joints_marker_array)

    def left_callback(self, msg: GloveDataMsg):
        self.base_callback(msg, "left")

    def right_callback(self, msg: GloveDataMsg):
        self.base_callback(msg, "right")

    def base_callback(self, msg: GloveDataMsg, hand_type: str):
        self.visualizer.update(msg.joint_angles, hand_type)

        if self.publish_joints:
            self.publish_joint_markers(msg, hand_type)

    def destroy_node(self):
        self.visualizer.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = VisualizerNode()
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