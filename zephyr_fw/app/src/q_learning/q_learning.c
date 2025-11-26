int reward(int state, int action)
{
//   int next_state=state+action;
//   if (next_state + 1 >= 20 || next_state + 2 >= 20) {
//     return -10;
//   } else if (next_state >= 20) {
//     return 20;
//   } else if (next_state % 3 == 2) {
//     return 10;
//   } else {
//     return -5;
//   }
}

int policy(int state)
{
    // true for only two possible actions
  int reward_1 = reward(state,1);
  int reward_2 = reward(state,2);
  if (reward_1 > reward_2) {
    return 1;
  }
  return 2;
}

float Q_value(int state, int action, float learning_rate)
{
  int next_state = state + action;
  // should always be running if in the correct state from state machine
//   if (state > 20) {
//     return 0;
//   }
  return reward(state, action) + learning_rate * Q_value(next_state, policy(next_state), learning_rate);
}
