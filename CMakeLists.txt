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
ament_python_install_package(${PROJECT_NAME})

install(
  DIRECTORY model
  DESTINATION share/${PROJECT_NAME}
)

install(
  PROGRAMS
    glove_ros/serial_node.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
