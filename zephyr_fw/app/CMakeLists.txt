cmake_minimum_required(VERSION 3.20.0)
find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(one_leg_robot_fw)

# Add include directories
zephyr_include_directories(include)

# Add all source files
target_sources(app PRIVATE
    src/main.c
    src/state_machine.c
    src/imu_reader.c
    src/leg_controller.c
    src/pid_controller.c
    src/safety_monitor.c
    src/serial_comm.c
)
