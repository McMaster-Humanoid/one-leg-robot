#include <zephyr/device.h>
#include <zephyr/drivers/gpio.h>
#include <zephyr/drivers/uart.h>
#include <zephyr/kernel.h>

#include "config.h"
#include "imu_reader.h"
#include "leg_controller.h"
#include "motor_control.h"
#include "pid_controller.h"
#include "safety_monitor.h"
#include "serial_comm.h"
#include "state_machine.h"

// ===============================================================
// THREAD PERIODS (tunable)
// ===============================================================
#define IMU_PERIOD_MS 1      // 1000 Hz
#define CONTROL_PERIOD_MS 1  // 1000 Hz
#define SERIAL_PERIOD_MS 5   // 200 Hz
#define SAFETY_PERIOD_MS 10  // 100 Hz

// ===============================================================
// THREAD STACKS & PRIORITIES
// (higher number = *lower* priority in Zephyr)
// ===============================================================

K_THREAD_STACK_DEFINE(imu_stack, 2048);
K_THREAD_STACK_DEFINE(control_stack, 2048);
K_THREAD_STACK_DEFINE(serial_stack, 2048);
K_THREAD_STACK_DEFINE(safety_stack, 2048);

static struct k_thread imu_thread;
static struct k_thread control_thread;
static struct k_thread serial_thread;
static struct k_thread safety_thread;

// Forward declarations
void imu_task(void*, void*, void*);
void control_task(void*, void*, void*);
void serial_task(void*, void*, void*);
void safety_task(void*, void*, void*);

// ===============================================================
// MAIN ENTRY POINT
// ===============================================================
void main(void) {
  printk("\n=== Humanoid Robot Firmware Booting ===\n");

  // -----------------------------------------------------------
  // 1. Hardware Initialization (MUST come first)
  // -----------------------------------------------------------
  init_hardware();  // GPIO, UART, timers, PWM, ADC, anything electrical

  // -----------------------------------------------------------
  // 2. Initialize all subsystem modules
  //    (NO module initializes inside another module!)
  // -----------------------------------------------------------
  init_imu_reader();
  init_motor_control();
  init_leg_controller();
  init_pid_controller();  // if needed
  init_state_machine();
  init_safety_monitor();
  init_serial_comm();

  printk("Initialization complete. Launching threads...\n");

  // -----------------------------------------------------------
  // 3. Create periodic real-time threads
  // -----------------------------------------------------------

  // IMU Thread (highest priority)
  k_thread_create(&imu_thread, imu_stack, K_THREAD_STACK_SIZEOF(imu_stack),
                  imu_task, NULL, NULL, NULL, 1, 0, K_NO_WAIT);

  // Control Thread (same high priority)
  k_thread_create(&control_thread, control_stack,
                  K_THREAD_STACK_SIZEOF(control_stack), control_task, NULL,
                  NULL, NULL, 1, 0, K_NO_WAIT);

  // Serial Comm Thread (medium priority)
  k_thread_create(&serial_thread, serial_stack,
                  K_THREAD_STACK_SIZEOF(serial_stack), serial_task, NULL, NULL,
                  NULL, 3, 0, K_NO_WAIT);

  // Safety Monitor Thread (high but less than control)
  k_thread_create(&safety_thread, safety_stack,
                  K_THREAD_STACK_SIZEOF(safety_stack), safety_task, NULL, NULL,
                  NULL, 2, 0, K_NO_WAIT);

  printk("All threads created.\n");

  // -----------------------------------------------------------
  // 4. MAIN LOOP (lowest priority)
  // -----------------------------------------------------------
  while (1) {
    // Heartbeat LED or low-frequency diagnostics
    k_sleep(K_MSEC(1000));
  }
}

// ===============================================================
// THREAD IMPLEMENTATIONS
// ===============================================================

void imu_task(void* a, void* b, void* c) {
  while (1) {
    imu_update();  // you will implement this in imu_reader.c
    k_sleep(K_MSEC(IMU_PERIOD_MS));
  }
}

void control_task(void* a, void* b, void* c) {
  while (1) {
    // Step 1: Get the desired torque from leg controller
    LegControlOutput cmd = leg_controller_update();

    // Step 2: Pass to motor control (which applies PID + PWM)
    motor_control_apply(cmd);

    k_sleep(K_MSEC(CONTROL_PERIOD_MS));
  }
}

void serial_task(void* a, void* b, void* c) {
  while (1) {
    serial_comm_update();
    k_sleep(K_MSEC(SERIAL_PERIOD_MS));
  }
}

void safety_task(void* a, void* b, void* c) {
  while (1) {
    safety_monitor_check();
    k_sleep(K_MSEC(SAFETY_PERIOD_MS));
  }
}
