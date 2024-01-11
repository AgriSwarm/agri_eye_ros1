#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

std::string exec(const char* cmd) {
    std::array<char, 128> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
    if (!pipe) {
        throw std::runtime_error("popen() failed!");
    }
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    return result;
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "local_sensing");
    ros::NodeHandle nh;
    image_transport::ImageTransport it(nh);

    ros::NodeHandle pnh("~");
    int camera_number;
    pnh.param("camera_number", camera_number, 0);
    
    image_transport::Publisher pub = it.advertise("/camera/image_raw", 1);

    std::string command = "v4l2-ctl --list-devices";
    try {
        std::string devicesList = exec(command.c_str());
        std::cout << "Avairable video devices:\n" << devicesList << std::endl;
    } catch (const std::runtime_error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    cv::VideoCapture cap(camera_number);
    if (!cap.isOpened()) {
        ROS_ERROR("Webcam cannot be opened.");
        return 1;
    }
    
    double fps = 30.0;
    ros::Rate loop_rate(fps);

    while (nh.ok()) {
        cv::Mat frame;
        if (!cap.read(frame)) {
            ROS_ERROR("Failed to capture frame");
            break;
        }
        sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", frame).toImageMsg();
        msg->header.stamp = ros::Time::now();
        pub.publish(msg);
        ros::spinOnce();
        loop_rate.sleep();
    }

    return 0;
}