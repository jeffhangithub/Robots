cmake_minimum_required(VERSION 3.8)
project(xsens_mvn_ros_msgs)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

# Create messages
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/LinkState.msg"
  "msg/LinkStateArray.msg"
  DEPENDENCIES std_msgs geometry_msgs
)

# Export dependencies
ament_export_dependencies(rosidl_default_runtime)
ament_export_dependencies(std_msgs)
ament_export_dependencies(geometry_msgs)

# Install messages
ament_export_include_directories(include)

# Install the package
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()