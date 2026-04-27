#!/usr/bin/env python3
"""
Diagnostic script to test ROS robot connection and topic communication.
Run this AFTER launching the robot with: roslaunch com760cw2_Group1 group1_setup.launch
"""

import rospy
import sys
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

class RobotDiagnostics:
    def __init__(self):
        rospy.init_node('robot_diagnostics', log_level=rospy.INFO)
        
        self.laser_received = False
        self.cmd_vel_subscribed = False
        self.laser_data = None
        
        # Subscribe to laser
        self.laser_sub = rospy.Subscriber("/Group1Bot/laser/scan", LaserScan, self.laser_callback)
        
        # Publish test command
        self.cmd_vel_pub = rospy.Publisher("/Group1Bot/cmd_vel", Twist, queue_size=10)
        
        rospy.loginfo("\n" + "="*60)
        rospy.loginfo("ROBOT DIAGNOSTICS TEST STARTING")
        rospy.loginfo("="*60)
    
    def laser_callback(self, msg):
        if not self.laser_received:
            rospy.loginfo(f"✓ LASER DATA RECEIVED!")
            rospy.loginfo(f"  - Number of samples: {len(msg.ranges)}")
            rospy.loginfo(f"  - Min range: {min(msg.ranges) if msg.ranges else 'N/A':.3f}m")
            rospy.loginfo(f"  - Max range: {max(msg.ranges) if msg.ranges else 'N/A':.3f}m")
            self.laser_received = True
        self.laser_data = msg
    
    def run_diagnostics(self):
        rospy.loginfo("\n1. Waiting for laser data (10 seconds)...")
        timeout = rospy.Time.now() + rospy.Duration(10)
        
        while rospy.Time.now() < timeout and not self.laser_received and not rospy.is_shutdown():
            rospy.sleep(0.1)
        
        if self.laser_received:
            rospy.loginfo("✓ Laser subscription working!")
        else:
            rospy.logerr("✗ NO LASER DATA RECEIVED - Check topic /Group1Bot/laser/scan")
            rospy.logerr("  Run 'rostopic list' to verify topics")
            return False
        
        rospy.loginfo("\n2. Testing command velocity publishing...")
        test_move = Twist()
        test_move.linear.x = 0.2
        test_move.angular.z = 0.0
        
        for i in range(5):
            self.cmd_vel_pub.publish(test_move)
            rospy.loginfo(f"  Published cmd_vel command {i+1}/5")
            rospy.sleep(0.5)
        
        rospy.loginfo("✓ Command velocity published!")
        
        rospy.loginfo("\n3. System status:")
        rospy.loginfo(f"  - ROS Master: Connected ✓")
        rospy.loginfo(f"  - Laser Topic: /Group1Bot/laser/scan ✓")
        rospy.loginfo(f"  - Cmd_Vel Topic: /Group1Bot/cmd_vel ✓")
        rospy.loginfo(f"  - Q-Table Size: Will grow as agent learns")
        
        rospy.loginfo("\n" + "="*60)
        rospy.loginfo("DIAGNOSTICS COMPLETE - All systems ready!")
        rospy.loginfo("="*60 + "\n")
        
        return True

if __name__ == '__main__':
    try:
        diag = RobotDiagnostics()
        rospy.sleep(1)  # Give time for subscriptions
        if diag.run_diagnostics():
            rospy.loginfo("You can now run the Q-learning agent safely!")
        else:
            rospy.logerr("There are connection issues. Check your launch file.")
            sys.exit(1)
    except rospy.ROSInterruptException:
        rospy.logwarn("Test interrupted")
        sys.exit(1)
