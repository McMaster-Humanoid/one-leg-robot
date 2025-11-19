#include <rclcpp/rclcpp.hpp>
#include <leg_controller/msg/remote_input.hpp>
#include <chrono>
#include <memory>

class JoystickReader : public rclcpp::Node
{
public:
    JoystickReader()
        : Node("joystick_reader")
    {

        remote_input_pub_ = this->create_publisher<leg_controller::msg::RemoteInput>(
            "/remote_input", 10);

        publish_timer_ = this->create_wall_timer(
            20ms, std::bind(&JoystickReader::publish_remote_input, this));

        timeout_timer_ = this->create_wall_timer(
            100ms, std::bind(&JoystickReader::check_timeout, this));

        current_input_.left_stick_y = 0.0;
        current_input_.right_stick_y = 0.0;
        current_input_.amplitude = 0.0;
        current_input_.frequency = 0.0;
        current_input_.mode_switch = 0;

        last_input_time_ = this->now();
        input_timeout_threshold_ = 500ms;

        RCLCPP_INFO(this->get_logger(), "JoystickReader initialized");
    }

    void process_input(double left_stick, double right_stick, int mode_button)
    {
        current_input_.left_stick_y = (left_stick + 1.0) / 2.0;
        current_input_.right_stick_y = (right_stick + 1.0) / 2.0;
        current_input_.mode_switch = mode_button;

        current_input_.amplitude = current_input_.left_stick_y;
        current_input_.frequency = current_input_.right_stick_y;

        last_input_time_ = this->now();
        input_received_ = true;
    }

private:
    void publish_remote_input()
    {
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

    rclcpp::Time last_input_time_;
    std::chrono::milliseconds input_timeout_threshold_;
    bool input_received_ = false;
};
