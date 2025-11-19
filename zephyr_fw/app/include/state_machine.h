#ifndef STATE_MACHINE_H
#define STATE_MACHINE_H

#include <stdbool.h>
#include <stdint.h>

// Low-level robot motion modes
typedef enum {
  MODE_INIT = 0,
  MODE_STAND,
  MODE_HOP,
  MODE_LAND,
  MODE_FAILSAFE
} RobotMode;

// External API (called ONLY from main.c)
void init_state_machine(void);

// Called by serial (ROS requests) OR other firmware modules
void state_machine_update(RobotMode request);

// Read current mode safely
RobotMode get_current_mode(void);

// Internal transitions (used by safety monitor)
void state_machine_force_failsafe(void);

#endif
