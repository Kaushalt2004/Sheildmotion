# ROS 2 node for integration with CARLA
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class CarlaRos2Bridge(Node):
    def __init__(self):
        super().__init__('carla_ros2_bridge')
        self.subscription = self.create_subscription(
            String,
            'carla/ego_vehicle/odometry',
            self.listener_callback,
            10)
        self.publisher = self.create_publisher(String, 'carla/commands', 10)

    def listener_callback(self, msg):
        self.get_logger().info(f'Received odometry: {msg.data}')
        # Example: publish a command
        cmd = String()
        cmd.data = 'drive'
        self.publisher.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = CarlaRos2Bridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
