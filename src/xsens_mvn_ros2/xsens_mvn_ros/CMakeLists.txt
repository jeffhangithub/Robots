cmake_minimum_required(VERSION 3.8)
project(xsens_mvn_ros)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 设置RPATH
set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib")
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(urdf REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(xsens_mvn_ros_msgs REQUIRED)
find_package(Boost REQUIRED COMPONENTS thread system chrono)

# Add include directories
include_directories(
  include
  ${EIGEN3_INCLUDE_DIRS}
  xsens_mvn_sdk/include
)

# Add subdirectory for SDK
add_subdirectory(xsens_mvn_sdk)

# Create library for HumanDataHandler
add_library(${PROJECT_NAME}_HumanDataHandler SHARED 
  src/xsens_client/HumanDataHandler.cpp
)

ament_target_dependencies(${PROJECT_NAME}_HumanDataHandler
  rclcpp
  geometry_msgs
  sensor_msgs
  std_msgs
  tf2_ros
  xsens_mvn_ros_msgs
  Eigen3
)

# Create library for XSensClient
add_library(${PROJECT_NAME}_XSensClient SHARED 
  src/xsens_client/XSensClient.cpp 
  src/xsens_client/Socket.cpp
)

ament_target_dependencies(${PROJECT_NAME}_XSensClient
  rclcpp
  geometry_msgs
  sensor_msgs
  std_msgs
  tf2_ros
  xsens_mvn_ros_msgs
  Eigen3
)

target_link_libraries(${PROJECT_NAME}_XSensClient 
  ${PROJECT_NAME}_HumanDataHandler 
  ${PROJECT_NAME}_SDK
  ${Boost_LIBRARIES}
)

# Create the main executable
add_executable(xsens_client_node src/xsens_client_node.cpp)
ament_target_dependencies(xsens_client_node
  rclcpp
  geometry_msgs
  sensor_msgs
  std_msgs
  tf2_ros
  tf2_geometry_msgs
  urdf
  Eigen3
  xsens_mvn_ros_msgs
  tf2_geometry_msgs
)

target_link_libraries(xsens_client_node
  ${PROJECT_NAME}_XSensClient
)

# Install targets
install(TARGETS
  xsens_client_node
  ${PROJECT_NAME}_HumanDataHandler
  ${PROJECT_NAME}_XSensClient
  DESTINATION lib/${PROJECT_NAME}
)

# Install headers
install(DIRECTORY include/
  DESTINATION include/
  PATTERN "*.h"
)

# Install launch files
install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

# Export dependencies
ament_export_include_directories(include)
ament_export_libraries(
  ${PROJECT_NAME}_HumanDataHandler
  ${PROJECT_NAME}_XSensClient
)
ament_export_dependencies(
  rclcpp
  geometry_msgs
  sensor_msgs
  std_msgs
  tf2_ros
  urdf
  Eigen3
  xsens_mvn_ros_msgs
)

# Install node executable
install(TARGETS
  xsens_client_node
  ${PROJECT_NAME}_HumanDataHandler 
  ${PROJECT_NAME}_SDK
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()