cmake_minimum_required(VERSION 3.5)
project(glove_ros)

find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)

find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/GloveDataMsg.msg"
  DEPENDENCIES std_msgs sensor_msgs builtin_interfaces
)

ament_export_dependencies(rosidl_default_runtime)

install(
  PROGRAMS
    glove_ros/serial_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME serial_node
)

install(
  PROGRAMS
    glove_ros/visualizer_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME visualizer_node
)

ament_package()