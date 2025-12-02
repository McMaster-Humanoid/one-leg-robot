#include <rclcpp/rclcpp.hpp>
#include <leg_controller/msg/remote_input.hpp>
#include <memory>
#include <cmath>

class JoystickTestNode : public rclcpp::Node
{
public:
    JoystickTestNode()
        : Node("joystick_test_node")
    {
        // Subscribe to the remote_input topic
        subscription_ = this->create_subscription<leg_controller::msg::RemoteInput>(
            "/remote_input", 10,
            std::bind(&JoystickTestNode::joystick_callback, this, std::placeholders::_1));

        // Initialize previous values for change detection
        prev_left_stick_ = -1.0;
        prev_right_stick_ = -1.0;
        prev_amplitude_ = -1.0;
        prev_frequency_ = -1.0;
        prev_mode_switch_ = -1;
        message_count_ = 0;
        last_print_time_ = this->now();

        RCLCPP_INFO(this->get_logger(), "Joystick test node started. Listening to /remote_input topic...");
        RCLCPP_INFO(this->get_logger(), "Displaying updates when values change (throttled to 2Hz)...");
    }

private:
    void joystick_callback(const leg_controller::msg::RemoteInput::SharedPtr msg)
    {
        message_count_++;
        
        const double change_threshold = 0.01;
        bool values_changed = 
            std::abs(msg->left_stick_y - prev_left_stick_) > change_threshold ||
            std::abs(msg->right_stick_y - prev_right_stick_) > change_threshold ||
            std::abs(msg->amplitude - prev_amplitude_) > change_threshold ||
            std::abs(msg->frequency - prev_frequency_) > change_threshold ||
            msg->mode_switch != prev_mode_switch_;

        auto now = this->now();
        auto time_since_last = now - last_print_time_;
        bool should_print = (values_changed && time_since_last.seconds() >= 0.5) || 
                           (message_count_ == 1);

        if (should_print)
        {
            // Print all the joystick data
            RCLCPP_INFO(this->get_logger(), 
                "=== Joystick Input (msg #%lu) ===", message_count_);
            RCLCPP_INFO(this->get_logger(), 
                "  Timestamp: %d.%09u", 
                msg->header.stamp.sec, msg->header.stamp.nanosec);
            RCLCPP_INFO(this->get_logger(), 
                "  Left Stick Y:  %.3f (amplitude)", msg->left_stick_y);
            RCLCPP_INFO(this->get_logger(), 
                "  Right Stick Y: %.3f (frequency)", msg->right_stick_y);
            RCLCPP_INFO(this->get_logger(), 
                "  Amplitude:     %.3f", msg->amplitude);
            RCLCPP_INFO(this->get_logger(), 
                "  Frequency:     %.3f", msg->frequency);
            RCLCPP_INFO(this->get_logger(), 
                "  Mode Switch:   %d", msg->mode_switch);
            
            int left_bar = static_cast<int>(msg->left_stick_y * 20);
            int right_bar = static_cast<int>(msg->right_stick_y * 20);
            std::string left_visual = std::string(left_bar, '=') + std::string(20 - left_bar, ' ');
            std::string right_visual = std::string(right_bar, '=') + std::string(20 - right_bar, ' ');
            RCLCPP_INFO(this->get_logger(), 
                "  Visual:        [%s] [%s]", left_visual.c_str(), right_visual.c_str());
            RCLCPP_INFO(this->get_logger(), 
                "=====================");

            prev_left_stick_ = msg->left_stick_y;
            prev_right_stick_ = msg->right_stick_y;
            prev_amplitude_ = msg->amplitude;
            prev_frequency_ = msg->frequency;
            prev_mode_switch_ = msg->mode_switch;
            last_print_time_ = now;
        }
    }

    rclcpp::Subscription<leg_controller::msg::RemoteInput>::SharedPtr subscription_;
    
    double prev_left_stick_;
    double prev_right_stick_;
    double prev_amplitude_;
    double prev_frequency_;
    int prev_mode_switch_;
    rclcpp::Time last_print_time_;
    size_t message_count_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<JoystickTestNode>());
    rclcpp::shutdown();
    return 0;
}

