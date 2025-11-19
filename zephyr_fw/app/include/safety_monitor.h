#ifndef SAFETY_MONITOR_H
#define SAFETY_MONITOR_H

#include <stdbool.h>

// Initialize safety monitor
void init_safety_monitor(void);

// Main safety check function (called by thread)
void safety_monitor_check(void);

// Enter failsafe mode
void enter_failsafe(void);

// Global fault flag
extern bool fault_detected;

#endif

