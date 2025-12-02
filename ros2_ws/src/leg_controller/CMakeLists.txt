cmake_minimum_required(VERSION 3.8)
project(leg_controller)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Generate messages
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/RemoteInput.msg"
  DEPENDENCIES std_msgs
)

# Add executable for joystick handler
add_executable(joystick_handler joystick_handler.cpp)
ament_target_dependencies(joystick_handler 
  rclcpp
)
rosidl_get_typesupport_target(cpp_typesupport_target
  ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(joystick_handler "${cpp_typesupport_target}")

# Add executable for joystick test node
add_executable(joystick_test_node joystick_test_node.cpp)
ament_target_dependencies(joystick_test_node 
  rclcpp
)
target_link_libraries(joystick_test_node "${cpp_typesupport_target}")

# Install the executables
install(TARGETS
  joystick_handler
  joystick_test_node
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
