#ifndef MOTOR_CONTROL_H
#define MOTOR_CONTROL_H

#include <stdint.h>

// Motor state structure
typedef struct {
    float position;   // encoder position
    float velocity;   // computed velocity
    float current;    // motor current in amperes
    float torque;     // motor torque
} MotorState;

// Leg control output structure (from leg_controller)
typedef struct {
    float desired_torque;
} LegControlOutput;

// Initialize motor control subsystem
void init_motor_control(void);

// Apply motor control command
void motor_control_apply(LegControlOutput cmd);

// Get current motor state
MotorState get_motor_state(void);

// Disable motors (for safety)
void disable_motors(void);

#endif

