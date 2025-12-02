#include <rclcpp/rclcpp.hpp>
#include <leg_controller/msg/remote_input.hpp>
#include <chrono>
#include <memory>
#include <thread>
#include <mutex>
#include <atomic>

#ifdef _WIN32
#include <conio.h>
#include <windows.h>
#else
#include <termios.h>
#include <unistd.h>
#include <fcntl.h>
#endif

class JoystickHandler : public rclcpp::Node
{
public:
    JoystickHandler()
        : Node("joystick_handler")
    {

        remote_input_pub_ = this->create_publisher<leg_controller::msg::RemoteInput>(
            "/remote_input", 10);

        publish_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(20), std::bind(&JoystickHandler::publish_remote_input, this));

        timeout_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(100), std::bind(&JoystickHandler::check_timeout, this));

        current_input_.left_stick_y = 0.0;
        current_input_.right_stick_y = 0.0;
        current_input_.amplitude = 0.0;
        current_input_.frequency = 0.0;
        current_input_.mode_switch = 0;

        last_input_time_ = this->now();
        input_timeout_threshold_ = std::chrono::milliseconds(500);

        // Initialize keyboard input
        setup_keyboard_input();
        
        // Start keyboard reading thread
        keyboard_thread_running_ = true;
        keyboard_thread_ = std::thread(&JoystickHandler::keyboard_input_thread, this);

        RCLCPP_INFO(this->get_logger(), "JoystickHandler initialized");
        RCLCPP_INFO(this->get_logger(), "Keyboard controls:");
        RCLCPP_INFO(this->get_logger(), "  W/S: Left stick Y (amplitude) - Up/Down");
        RCLCPP_INFO(this->get_logger(), "  A/D: Right stick Y (frequency) - Left/Right");
        RCLCPP_INFO(this->get_logger(), "  Space: Mode switch toggle");
        RCLCPP_INFO(this->get_logger(), "  Q: Quit");
    }

    ~JoystickHandler()
    {
        keyboard_thread_running_ = false;
        if (keyboard_thread_.joinable())
        {
            keyboard_thread_.join();
        }
        restore_keyboard_input();
    }

    void process_input(double left_stick, double right_stick, int mode_button)
    {
        std::lock_guard<std::mutex> lock(input_mutex_);
        current_input_.left_stick_y = std::max(0.0, std::min(1.0, (left_stick + 1.0) / 2.0));
        current_input_.right_stick_y = std::max(0.0, std::min(1.0, (right_stick + 1.0) / 2.0));
        current_input_.mode_switch = mode_button;

        current_input_.amplitude = current_input_.left_stick_y;
        current_input_.frequency = current_input_.right_stick_y;

        last_input_time_ = this->now();
        input_received_ = true;
    }

    bool should_quit() const { return should_quit_; }

private:
    void setup_keyboard_input()
    {
#ifdef _WIN32
        HANDLE hStdin = GetStdHandle(STD_INPUT_HANDLE);
        DWORD mode;
        GetConsoleMode(hStdin, &mode);
        SetConsoleMode(hStdin, mode & ~(ENABLE_LINE_INPUT | ENABLE_ECHO_INPUT));
#else
        // Check if stdin is a TTY
        if (!isatty(STDIN_FILENO))
        {
            RCLCPP_WARN(this->get_logger(), 
                "stdin is not a TTY. Keyboard input may not work. Make sure to run in a terminal.");
        }
        
        struct termios term;
        if (tcgetattr(STDIN_FILENO, &old_term_) != 0)
        {
            RCLCPP_ERROR(this->get_logger(), "Failed to get terminal attributes");
            return;
        }
        term = old_term_;
        term.c_lflag &= ~(ICANON | ECHO);
        term.c_cc[VMIN] = 0;
        term.c_cc[VTIME] = 0;
        if (tcsetattr(STDIN_FILENO, TCSANOW, &term) != 0)
        {
            RCLCPP_ERROR(this->get_logger(), "Failed to set terminal attributes");
        }
        
        int flags = fcntl(STDIN_FILENO, F_GETFL);
        if (flags == -1)
        {
            RCLCPP_ERROR(this->get_logger(), "Failed to get file flags");
        }
        else
        {
            if (fcntl(STDIN_FILENO, F_SETFL, flags | O_NONBLOCK) == -1)
            {
                RCLCPP_ERROR(this->get_logger(), "Failed to set non-blocking mode");
            }
        }
        
        RCLCPP_INFO(this->get_logger(), "Terminal configured for keyboard input. Make sure this terminal has focus!");
#endif
    }

    void restore_keyboard_input()
    {
#ifndef _WIN32
        tcsetattr(STDIN_FILENO, TCSANOW, &old_term_);
#endif
    }

    bool is_key_pressed(int virtual_key)
    {
#ifdef _WIN32
        return (GetAsyncKeyState(virtual_key) & 0x8000) != 0;
#else
        // For Linux, we'd need a different approach
        return false;
#endif
    }

    int get_key()
    {
#ifdef _WIN32
        if (_kbhit())
        {
            return _getch();
        }
        return 0;
#else
        char ch;
        if (read(STDIN_FILENO, &ch, 1) == 1)
        {
            return ch;
        }
        return 0;
#endif
    }

    void keyboard_input_thread()
    {
        double left_stick = 0.0;   // -1.0 to 1.0
        double right_stick = 0.0;  // -1.0 to 1.0
        int mode_switch = 0;
        bool mode_was_pressed = false;
        bool q_was_pressed = false;

#ifdef _WIN32
        // Windows: Use GetAsyncKeyState for real-time key state
        while (keyboard_thread_running_ && rclcpp::ok())
        {
            bool w_pressed = is_key_pressed('W');
            bool s_pressed = is_key_pressed('S');
            bool a_pressed = is_key_pressed('A');
            bool d_pressed = is_key_pressed('D');
            bool space_pressed = is_key_pressed(VK_SPACE);
            bool q_pressed = is_key_pressed('Q');

            // Update stick values based on currently pressed keys
            double new_left_stick = 0.0;
            if (w_pressed && !s_pressed)
            {
                new_left_stick = 1.0;  // Full up
            }
            else if (s_pressed && !w_pressed)
            {
                new_left_stick = -1.0;  // Full down
            }
            
            double new_right_stick = 0.0;
            if (a_pressed && !d_pressed)
            {
                new_right_stick = -1.0;  // Full left
            }
            else if (d_pressed && !a_pressed)
            {
                new_right_stick = 1.0;  // Full right
            }

            if (space_pressed && !mode_was_pressed)
            {
                mode_switch = (mode_switch == 0) ? 1 : 0;
                RCLCPP_INFO(this->get_logger(), "Mode switch: %d", mode_switch);
            }
            mode_was_pressed = space_pressed;

            // Quit - Q key
            if (q_pressed && !q_was_pressed)
            {
                RCLCPP_INFO(this->get_logger(), "Quit requested");
                should_quit_ = true;
                rclcpp::shutdown();
                break;
            }
            q_was_pressed = q_pressed;

            // Update if values changed
            if (new_left_stick != left_stick || new_right_stick != right_stick)
            {
                left_stick = new_left_stick;
                right_stick = new_right_stick;
                process_input(left_stick, right_stick, mode_switch);
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
#else
        // Linux: Track key states manually (terminal input doesn't support key hold detection)
        bool w_active = false;
        bool s_active = false;
        bool a_active = false;
        bool d_active = false;
        auto last_key_time = std::chrono::steady_clock::now();
        const auto key_hold_duration = std::chrono::milliseconds(150);  // Consider key "held" for 150ms
        size_t key_press_count = 0;
        bool first_key_received = false;

        while (keyboard_thread_running_ && rclcpp::ok())
        {
            int key = get_key();
            auto current_time = std::chrono::steady_clock::now();
            
            // Process key press events
            if (key != 0)
            {
                if (!first_key_received)
                {
                    RCLCPP_INFO(this->get_logger(), "Keyboard input detected! Key code: %d", key);
                    first_key_received = true;
                }
                
                key_press_count++;
                last_key_time = current_time;
                
                if (key == 'w' || key == 'W')
                {
                    w_active = true;
                    s_active = false;
                }
                else if (key == 's' || key == 'S')
                {
                    s_active = true;
                    w_active = false;
                }
                else if (key == 'a' || key == 'A')
                {
                    a_active = true;
                    d_active = false;
                }
                else if (key == 'd' || key == 'D')
                {
                    d_active = true;
                    a_active = false;
                }
                else if (key == ' ')
                {
                    if (!mode_was_pressed)
                    {
                        mode_switch = (mode_switch == 0) ? 1 : 0;
                        RCLCPP_INFO(this->get_logger(), "Mode switch: %d", mode_switch);
                    }
                    mode_was_pressed = true;
                }
                else if (key == 'q' || key == 'Q')
                {
                    if (!q_was_pressed)
                    {
                        RCLCPP_INFO(this->get_logger(), "Quit requested");
                        should_quit_ = true;
                        rclcpp::shutdown();
                        break;
                    }
                    q_was_pressed = true;
                }
                else if (key == '\n' || key == '\r')
                {
                    // Ignore enter key
                }
                else
                {
                    // Unknown key - log it for debugging
                    RCLCPP_DEBUG_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                        "Unknown key pressed: %d (0x%02x)", key, key);
                }
            }
            else
            {
                mode_was_pressed = false;
                q_was_pressed = false;
            }

            // Auto-release keys if no input for a while (simulates key release)
            auto time_since_key = std::chrono::duration_cast<std::chrono::milliseconds>(
                current_time - last_key_time);
            if (time_since_key > key_hold_duration)
            {
                if (w_active || s_active || a_active || d_active)
                {
                    w_active = false;
                    s_active = false;
                    a_active = false;
                    d_active = false;
                }
            }

            // Update stick values based on active keys
            double new_left_stick = 0.0;
            if (w_active)
            {
                new_left_stick = 1.0;  // Full up
            }
            else if (s_active)
            {
                new_left_stick = -1.0;  // Full down
            }
            
            double new_right_stick = 0.0;
            if (a_active)
            {
                new_right_stick = -1.0;  // Full left
            }
            else if (d_active)
            {
                new_right_stick = 1.0;  // Full right
            }

            // Update if values changed
            if (new_left_stick != left_stick || new_right_stick != right_stick)
            {
                left_stick = new_left_stick;
                right_stick = new_right_stick;
                process_input(left_stick, right_stick, mode_switch);
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        
        RCLCPP_INFO(this->get_logger(), "Keyboard input thread exiting. Total key presses: %zu", key_press_count);
#endif
    }

    void publish_remote_input()
    {
        std::lock_guard<std::mutex> lock(input_mutex_);
        auto time_since_input = this->now() - last_input_time_;
        if (time_since_input > input_timeout_threshold_)
        {
            current_input_.amplitude = 0.0;
            current_input_.frequency = 0.0;
            current_input_.left_stick_y = 0.0;
            current_input_.right_stick_y = 0.0;
            current_input_.mode_switch = 0;
        }

        current_input_.header.stamp = this->now();
        current_input_.header.frame_id = "remote_input";

        remote_input_pub_->publish(current_input_);
    }

    void check_timeout()
    {
        std::lock_guard<std::mutex> lock(input_mutex_);
        auto time_since_input = this->now() - last_input_time_;
        if (time_since_input > input_timeout_threshold_ && input_received_)
        {
            RCLCPP_WARN_THROTTLE(
                this->get_logger(), *this->get_clock(), 2000,
                "No joystick input received for >500ms, using safe defaults");
            input_received_ = false;
        }
    }

    rclcpp::Publisher<leg_controller::msg::RemoteInput>::SharedPtr remote_input_pub_;

    rclcpp::TimerBase::SharedPtr publish_timer_;
    rclcpp::TimerBase::SharedPtr timeout_timer_;

    leg_controller::msg::RemoteInput current_input_;
    std::mutex input_mutex_;

    rclcpp::Time last_input_time_;
    std::chrono::milliseconds input_timeout_threshold_;
    bool input_received_ = false;

    std::thread keyboard_thread_;
    std::atomic<bool> keyboard_thread_running_;
    std::atomic<bool> should_quit_{false};

#ifndef _WIN32
    struct termios old_term_;
#endif
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    
    auto handler = std::make_shared<JoystickHandler>();
    
    rclcpp::spin(handler);
    
    rclcpp::shutdown();
    return 0;
}