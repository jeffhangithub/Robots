#include <memory>
#include <string>
#include <map>

#include "rclcpp/rclcpp.hpp"
#include "tf2_ros/transform_broadcaster.h"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "xsens_mvn_ros_msgs/msg/link_state_array.hpp"
#include "geometry_msgs/msg/point.hpp"

#include "xsens_mvn_ros/XSensClient.h"

class XSensClientNode : public rclcpp::Node
{
public:
    XSensClientNode() : Node("xsens_client")
    {
        // Declare parameters
        this->declare_parameter<std::string>("model_name", "skeleton");
        this->declare_parameter<std::string>("reference_frame", "world");
        this->declare_parameter<int>("udp_port", 8001);

        // Get parameters
        model_name_ = this->get_parameter("model_name").as_string();
        reference_frame_ = this->get_parameter("reference_frame").as_string();
        int xsens_udp_port = this->get_parameter("udp_port").as_int();

        // Initialize transform broadcaster
        tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);

        // Create publishers
        joint_state_publisher_ = this->create_publisher<sensor_msgs::msg::JointState>("joint_states", 10);
        link_state_publisher_ = this->create_publisher<xsens_mvn_ros_msgs::msg::LinkStateArray>("link_states", 10);
        com_publisher_ = this->create_publisher<geometry_msgs::msg::Point>("com", 10);

        // Initialize XSens client
        try
        {
            xsens_client_ptr_ = std::make_shared<XSensClient>(xsens_udp_port);
        }
        catch(const std::exception& err)
        {
            RCLCPP_ERROR(this->get_logger(), "Failed to create XSens client: %s", err.what());
            throw;
        }
        
        if(!xsens_client_ptr_->init())
        {
            RCLCPP_ERROR(this->get_logger(), "XSens client initialization failed.");
            throw std::runtime_error("XSens client initialization failed");
        }

        // Create timer for main loop
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(8), // ~120Hz
            std::bind(&XSensClientNode::publishData, this));
    }

private:
    void publishData()
    {
        auto now = this->now();

        // Publish joint state
        if (joint_state_publisher_->get_subscription_count() > 0)
        {
            auto joint_state_msg = std::make_shared<sensor_msgs::msg::JointState>();
            joint_state_msg->header.stamp = now;

            auto joints = xsens_client_ptr_->getHumanData()->getJoints();
            for (auto& joint_pair : joints)
            {
                joint_state_msg->name.push_back(model_name_ + "_" + joint_pair.first + "_x");
                joint_state_msg->name.push_back(model_name_ + "_" + joint_pair.first + "_y");
                joint_state_msg->name.push_back(model_name_ + "_" + joint_pair.first + "_z");
                
                // Convert degrees to radians
                joint_state_msg->position.push_back(joint_pair.second.state.angles[0] / 180.0 * M_PI);
                joint_state_msg->position.push_back(joint_pair.second.state.angles[1] / 180.0 * M_PI);
                joint_state_msg->position.push_back(joint_pair.second.state.angles[2] / 180.0 * M_PI);
            }
            
            joint_state_publisher_->publish(*joint_state_msg);
        }
        
        // Publish link tf and state
        auto link_state_msg = std::make_shared<xsens_mvn_ros_msgs::msg::LinkStateArray>();
        auto links = xsens_client_ptr_->getHumanData()->getLinks();
        
        for (auto& link_pair : links)
        {
            // Publish link tf
            if (!(link_pair.second.state.orientation.x() == 0 && 
                  link_pair.second.state.orientation.y() == 0 && 
                  link_pair.second.state.orientation.z() == 0 && 
                  link_pair.second.state.orientation.w() == 0))
            {
                geometry_msgs::msg::TransformStamped transform_stamped;
                transform_stamped.header.stamp = now;
                transform_stamped.header.frame_id = reference_frame_;
                transform_stamped.child_frame_id = model_name_ + "_" + link_pair.first;
                
                transform_stamped.transform.translation.x = link_pair.second.state.position[0];
                transform_stamped.transform.translation.y = link_pair.second.state.position[1];
                transform_stamped.transform.translation.z = link_pair.second.state.position[2];
                
                transform_stamped.transform.rotation.x = link_pair.second.state.orientation.x();
                transform_stamped.transform.rotation.y = link_pair.second.state.orientation.y();
                transform_stamped.transform.rotation.z = link_pair.second.state.orientation.z();
                transform_stamped.transform.rotation.w = link_pair.second.state.orientation.w();
                
                tf_broadcaster_->sendTransform(transform_stamped);
            }

            if (link_state_publisher_->get_subscription_count() > 0)
            {
                // Publish link state
                xsens_mvn_ros_msgs::msg::LinkState link_state;
                link_state.header.frame_id = link_pair.first;
                link_state.header.stamp = now;
                
                // Convert Eigen types to geometry_msgs
                link_state.pose.position.x = link_pair.second.state.position[0];
                link_state.pose.position.y = link_pair.second.state.position[1];
                link_state.pose.position.z = link_pair.second.state.position[2];
                
                link_state.pose.orientation.x = link_pair.second.state.orientation.x();
                link_state.pose.orientation.y = link_pair.second.state.orientation.y();
                link_state.pose.orientation.z = link_pair.second.state.orientation.z();
                link_state.pose.orientation.w = link_pair.second.state.orientation.w();
                
                // Convert twist
                link_state.twist.linear.x = link_pair.second.state.velocity.linear[0];
                link_state.twist.linear.y = link_pair.second.state.velocity.linear[1];
                link_state.twist.linear.z = link_pair.second.state.velocity.linear[2];
                link_state.twist.angular.x = link_pair.second.state.velocity.angular[0];
                link_state.twist.angular.y = link_pair.second.state.velocity.angular[1];
                link_state.twist.angular.z = link_pair.second.state.velocity.angular[2];
                
                // Convert acceleration
                link_state.accel.linear.x = link_pair.second.state.acceleration.linear[0];
                link_state.accel.linear.y = link_pair.second.state.acceleration.linear[1];
                link_state.accel.linear.z = link_pair.second.state.acceleration.linear[2];
                link_state.accel.angular.x = link_pair.second.state.acceleration.angular[0];
                link_state.accel.angular.y = link_pair.second.state.acceleration.angular[1];
                link_state.accel.angular.z = link_pair.second.state.acceleration.angular[2];

                link_state_msg->states.push_back(link_state);
            }
        }

        if (link_state_publisher_->get_subscription_count() > 0)
        {
            link_state_publisher_->publish(*link_state_msg);
        }

        // Publish center of mass
        if (com_publisher_->get_subscription_count() > 0)
        {
            auto com_msg = std::make_shared<geometry_msgs::msg::Point>();
            auto com = xsens_client_ptr_->getHumanData()->getCOM();
            com_msg->x = com[0];
            com_msg->y = com[1];
            com_msg->z = com[2];
            com_publisher_->publish(*com_msg);
        }
    }

    std::shared_ptr<XSensClient> xsens_client_ptr_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_state_publisher_;
    rclcpp::Publisher<xsens_mvn_ros_msgs::msg::LinkStateArray>::SharedPtr link_state_publisher_;
    rclcpp::Publisher<geometry_msgs::msg::Point>::SharedPtr com_publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
    
    std::string model_name_;
    std::string reference_frame_;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    try
    {
        auto node = std::make_shared<XSensClientNode>();
        rclcpp::spin(node);
    }
    catch(const std::exception& e)
    {
        RCLCPP_ERROR(rclcpp::get_logger("xsens_client"), "Exception: %s", e.what());
        return -1;
    }
    rclcpp::shutdown();
    return 0;
}