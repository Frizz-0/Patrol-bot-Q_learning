#!/usr/bin/env python3
"""
Waypoint Finder Helper — Gazebo Classic (ROS Noetic)
COM760 CW2 Group 1

HOW TO USE:
  1. Start the hospital simulation:
       roslaunch com760cw2_group1 train_phase1.launch
  2. In a second terminal, run:
       rosrun com760cw2_group1 find_waypoints.py
  3. Use the Gazebo GUI to physically move the robot to a room/corridor
     you want as a patrol waypoint (drag it with the translation tool,
     or just note where it spawns after each episode reset).
  4. Press ENTER in this terminal to record the current robot position.
  5. Repeat for each waypoint (8 waypoints recommended for a full patrol).
  6. Copy the printed PATROL_ROUTE into train_phase2_waypoint_nav.py
     and train_phase3_patrol.py (the PATROL_ROUTE list near the top).

ALTERNATIVE — move the target_marker instead of the robot:
  In Gazebo, select the target_marker cylinder, drag it to a room,
  press ENTER here to record that position.
"""
import rospy, sys
from gazebo_msgs.srv import GetModelState

def main():
    rospy.init_node('waypoint_finder', anonymous=True)
    get_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
    rospy.wait_for_service('/gazebo/get_model_state', timeout=10)

    print("\n=== WAYPOINT FINDER ===")
    print("Press ENTER to record robot position, 'q'+ENTER to quit.\n")

    waypoints = []
    while not rospy.is_shutdown():
        try:
            key = input(f"[WP {len(waypoints)}] Press ENTER to record (q to quit): ")
        except (EOFError, KeyboardInterrupt):
            break

        if key.strip().lower() == 'q':
            break

        try:
            res = get_state('Group1Bot', 'world')
            x = round(res.pose.position.x, 2)
            y = round(res.pose.position.y, 2)
            waypoints.append((x, y))
            print(f"  ✓ Recorded WP{len(waypoints)-1}: ({x}, {y})")
        except Exception as e:
            print(f"  ✗ Failed to get robot position: {e}")

    if not waypoints:
        print("\nNo waypoints recorded.")
        return

    print("\n" + "="*50)
    print("Copy this into train_phase2_waypoint_nav.py")
    print("and train_phase3_patrol.py:\n")
    print("PATROL_ROUTE = [")
    for i, (x, y) in enumerate(waypoints):
        print(f"    ({x:6.1f}, {y:6.1f}),   # WP{i}")
    print("]")
    print("="*50)


if __name__ == '__main__':
    main()
