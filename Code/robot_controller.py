import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

class RobotHandROS2Controller(Node):
    def __init__(self):
        super().__init__('robot_hand_controller')
        self.joint_names = [
            "thumb_joint1", "thumb_joint2",
            "index_joint1", "index_joint2",
            "middle_joint1", "middle_joint2",
            "ring_joint1", "ring_joint2",
            "pinky_joint1", "pinky_joint2"
        ]
        self.pub = self.create_publisher(JointState, '/robot_hand/joint_states', 10)

    def publish_features(self, features):
        joint_angles = features.get('joint_angles', {})
        angles = []
        for finger in ["thumb", "index", "middle", "ring", "pinky"]:
            finger_angles = joint_angles.get(f"{finger}_angles", [0.0, 0.0])
            angles.extend(finger_angles)
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = angles
        self.pub.publish(msg)

def main():
    rclpy.init()
    node = RobotHandROS2Controller()
    example_features = {
        'joint_angles': {
            'thumb_angles': [0.5, 0.7],
            'index_angles': [0.6, 0.8],
            'middle_angles': [0.4, 0.9],
            'ring_angles': [0.3, 0.6],
            'pinky_angles': [0.2, 0.5]
        }
    }
    rate = node.create_rate(10)
    try:
        while rclpy.ok():
            node.publish_features(example_features)
            rate.sleep()
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()

if __name__ == "__main__":
    main()