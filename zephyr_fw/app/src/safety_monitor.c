#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>
#include <math.h>

#include "safety_monitor.h"
#include "motor_control.h"
#include "imu_reader.h"
#include "state_machine.h"

// Safety thresholds
#define MAX_SAFE_TILT 45.0f      // degrees
#define MAX_SAFE_CURRENT 5.0f    // amperes

// Watchdog timeout
#define WATCHDOG_TIMEOUT_MS 500

// Global fault flag
bool fault_detected = false;

// Watchdog timer
static struct k_timer watchdog_timer;

// Watchdog callback (called when timer expires)
void watchdog_expired(struct k_timer *timer) {
    printk("Watchdog timeout!\n");
    fault_detected = true;
    enter_failsafe();
}

void init_safety_monitor(void) {
    k_timer_init(&watchdog_timer, watchdog_expired, NULL);
    k_timer_start(&watchdog_timer, K_MSEC(WATCHDOG_TIMEOUT_MS), K_MSEC(WATCHDOG_TIMEOUT_MS));
}

void safety_monitor_check(void) {
    // Get IMU data (placeholder - assumes these functions exist)
    IMUData imu = get_imu_data();
    
    // Get motor state (placeholder - assumes these functions exist)
    MotorState motor = get_motor_state();
    
    // Check safety conditions
    if (fabs(imu.pitch) > MAX_SAFE_TILT ||
        fabs(motor.current) > MAX_SAFE_CURRENT) {
        fault_detected = true;
        enter_failsafe();
    }
    
    // Feed the watchdog to prevent timeout
    k_timer_start(&watchdog_timer, K_MSEC(WATCHDOG_TIMEOUT_MS), K_MSEC(WATCHDOG_TIMEOUT_MS));
}

void enter_failsafe(void) {
    // Disable motors (placeholder - assumes this function exists)
    disable_motors();
    
    // Update state machine to failsafe mode
    state_machine_update(MODE_FAILSAFE);
    
    // Log error
    printk("Failsafe triggered!\n");
}

