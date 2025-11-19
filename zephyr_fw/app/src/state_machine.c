#include "state_machine.h"

#include <zephyr/kernel.h>

static RobotMode current_mode = MODE_INIT;

// ============================================================
// INITIALIZATION
// ============================================================
void init_state_machine(void) { current_mode = MODE_INIT; }

// ============================================================
// INTERNAL HELPER FOR SAFE TRANSITIONS
// ============================================================
static void transition_to(RobotMode new_mode) {
  // Optional: add logging later
  // printk("STATE: %d -> %d\n", current_mode, new_mode);
  current_mode = new_mode;
}

// ============================================================
// MAIN STATE MACHINE UPDATE
// (Called when: serial commands OR internal triggers occur)
// ============================================================
void state_machine_update(RobotMode request) {
  switch (current_mode) {
    case MODE_INIT:
      if (request == MODE_STAND) {
        transition_to(MODE_STAND);
      }
      break;

    case MODE_STAND:
      if (request == MODE_HOP) {
        transition_to(MODE_HOP);
      } else if (request == MODE_INIT) {
        transition_to(MODE_INIT);
      }
      break;

    case MODE_HOP:
      // Landing is automatically detected by IMU in leg controller
      if (request == MODE_LAND) {
        transition_to(MODE_LAND);
      }
      break;

    case MODE_LAND:
      if (request == MODE_STAND) {
        transition_to(MODE_STAND);
      }
      break;

    case MODE_FAILSAFE:
      // stays locked unless INIT explicitly sent
      if (request == MODE_INIT) {
        transition_to(MODE_INIT);
      }
      break;

    default:
      transition_to(MODE_FAILSAFE);
      break;
  }
}

// ============================================================
// FORCE FAILSAFE (from safety_monitor.c)
// ============================================================
void state_machine_force_failsafe(void) { transition_to(MODE_FAILSAFE); }

// ============================================================
// GET CURRENT MODE
// ============================================================
RobotMode get_current_mode(void) { return current_mode; }
