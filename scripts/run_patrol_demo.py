#!/usr/bin/env python3
"""
Hospital Patrol Bot — Deployment Demo
COM760 CW2 Group 1

This is the FINAL PRODUCT. Loads the trained Q-table and runs the patrol
bot in pure exploitation mode (no random actions). The robot:

  1. Patrols every waypoint in the hospital in sequence
  2. Uses its camera to detect the green anomaly marker
  3. Stops and raises an alert when anomaly is found
  4. Resumes patrol after reporting (configurable)

PREREQUISITE: All three training phases complete.
              ~/q_table_p3.pkl must exist (falls back to p2, then p1).

RUN:
  roslaunch com760cw2_group1 run_patrol_demo.launch

or without a launch file:
  roslaunch com760cw2_group1 train_phase3.launch   ← start Gazebo + robot
  rosrun com760cw2_group1 run_patrol_demo.py        ← in a second terminal
"""
import os, rospy, numpy as np, random, math, pickle
from collections import deque
from sensor_msgs.msg import LaserScan, Image
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from std_srvs.srv import Empty
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState, GetModelState
from tf.transformations import euler_from_quaternion
from cv_bridge import CvBridge
import cv2

NUM_ACTIONS = 5

PATROL_ROUTE = [
    ( 0.0, 10.0),   # WP0 - spawn / north main corridor
    (-5.0,  7.0),   # WP1 - west waiting area
    ( 0.0,  3.0),   # WP2 - front lobby / nurses station
    ( 5.0,  7.0),   # WP3 - east waiting area
    ( 0.0, 16.0),   # WP4 - north corridor (toward elevators)
    ( 0.0, -4.0),   # WP5 - south junction
    (-8.0,-12.0),   # WP6 - south-west patient wing
    ( 8.0,-17.0),   # WP7 - south-east patient wing
]

WP_REACH_DIST    = 1.5
ANOMALY_DETECT_D = 2.5
ALERT_HOLD_SECS  = 5.0   # how long to stop at anomaly before resuming patrol

# Try trained files in order of quality
Q_TABLE_CANDIDATES = [
    os.path.expanduser('~/q_table_p3.pkl'),
    os.path.expanduser('~/q_table_p2.pkl'),
    os.path.expanduser('~/q_table_p1.pkl'),
]


class PatrolDemo:
    def __init__(self):
        rospy.init_node('patrol_demo_node')

        self.current_pos  = np.array([0.0, 0.0])
        self.robot_yaw    = 0.0
        self.pos_history  = deque(maxlen=80)
        self.stuck_count  = 0

        self.target_visible = False
        self.visual_error   = 0.0

        self.patrol_idx      = 0
        self.current_target  = np.array(PATROL_ROUTE[0])
        self.anomaly_pos     = np.array([7.0, 9.0])   # default from world file

        self.patrols_completed = 0
        self.anomalies_found   = 0
        self.total_wps_visited = 0

        self.alert_active      = False
        self.alert_start       = None
        # After alert clears, ignore detections for this many seconds so the
        # robot can move away before the camera triggers again.
        self.detection_cooldown_until = rospy.Time(0)

        # Pure exploitation — no random actions
        self.epsilon = 0.0

        self.q_table = {}
        self._load_best_q_table()

        self.bridge = CvBridge()

        self.get_model_proxy = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        self.set_state_proxy = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

        self.vel_pub   = rospy.Publisher('/Group1Bot/cmd_vel',    Twist,  queue_size=10)
        self.alert_pub = rospy.Publisher('/patrol/anomaly_alert', String, queue_size=10)

        rospy.Subscriber('/Group1Bot/camera/image_raw', Image,     self._image_cb)
        rospy.Subscriber('/Group1Bot/laser/scan',       LaserScan, self._laser_cb)

        self._log_banner()

    # ── banner ─────────────────────────────────────────────────────────────
    def _log_banner(self):
        rospy.logwarn(
            "\n╔══════════════════════════════════════════════╗\n"
            "║   HOSPITAL PATROL BOT — LIVE DEPLOYMENT      ║\n"
            "║   Mode    : Pure exploitation (ε = 0)        ║\n"
            f"║   Q-states: {len(self.q_table):<6d}                          ║\n"
            f"║   Waypoints: {len(PATROL_ROUTE)} hospital zones              ║\n"
            "║   Anomaly : detected by camera (green marker)║\n"
            "╚══════════════════════════════════════════════╝"
        )

    # ── Q-table loader ─────────────────────────────────────────────────────
    def _load_best_q_table(self):
        for path in Q_TABLE_CANDIDATES:
            if os.path.exists(path):
                try:
                    d   = pickle.load(open(path, 'rb'))
                    raw = d.get('q_table', {})
                    self.q_table = {
                        k: (v + [0.0]*(NUM_ACTIONS-len(v)))[:NUM_ACTIONS]
                        for k, v in raw.items()
                    }
                    eps = d.get('epsilon', 0)
                    ep  = d.get('episodes', 0)
                    rospy.loginfo(
                        f"[DEMO] Loaded {len(self.q_table)} Q-states from {path}\n"
                        f"       Trained for {ep} episodes | final ε={eps:.3f}"
                    )
                    return
                except Exception as e:
                    rospy.logwarn(f"[DEMO] Could not load {path}: {e}")
        rospy.logerr("[DEMO] No trained Q-table found! Run training phases first.")

    # ── camera ─────────────────────────────────────────────────────────────
    def _image_cb(self, msg):
        try:
            hsv  = cv2.cvtColor(self.bridge.imgmsg_to_cv2(msg, 'bgr8'),
                                cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, np.array([40, 80, 80]), np.array([80, 255, 255]))
            M    = cv2.moments(mask)
            if M['m00'] > 300:
                self.visual_error   = (int(M['m10']/M['m00']) - 320) / 320.0
                self.target_visible = True
            else:
                self.target_visible = False
                self.visual_error   = 0.0
        except Exception:
            pass

    # ── utils ──────────────────────────────────────────────────────────────
    def _safe_min(self, ranges):
        v = [x for x in ranges if math.isfinite(x) and x > 0.01]
        return min(v) if v else float('inf')

    def _ground_truth(self):
        try:
            res = self.get_model_proxy('Group1Bot', 'world')
            self.current_pos = np.array([res.pose.position.x, res.pose.position.y])
            q = res.pose.orientation
            _, _, self.robot_yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        except Exception:
            pass

    def _get_anomaly_pos(self):
        try:
            res = self.get_model_proxy('target_marker', 'world')
            self.anomaly_pos = np.array([res.pose.position.x, res.pose.position.y])
        except Exception:
            pass

    def _argmax_tie(self, q):
        m = max(q)
        return random.choice([i for i, v in enumerate(q) if v == m])

    def _default_q(self):
        return [0.0] * NUM_ACTIONS

    # ── state ──────────────────────────────────────────────────────────────
    def _state(self, msg, dist):
        sector_starts = [0, 144, 288, 432, 576]
        def level(r):
            d = self._safe_min(r)
            return 0 if d < 0.5 else (1 if d < 1.2 else 2)
        ls = tuple(level(msg.ranges[s:s+144]) for s in sector_starts)

        ang = math.atan2(self.current_target[1] - self.current_pos[1],
                         self.current_target[0] - self.current_pos[0])
        err = ang - self.robot_yaw
        while err >  math.pi: err -= 2*math.pi
        while err < -math.pi: err += 2*math.pi
        hd = int(((err + math.pi) / (2*math.pi)) * 8) % 8

        dist_bin = min(int(dist), 7)
        cam = 0
        if self.target_visible:
            cam = 1 if self.visual_error < -0.25 else (3 if self.visual_error > 0.25 else 2)

        return str(ls + (hd, dist_bin, cam))

    # ── anomaly alert ───────────────────────────────────────────────────────
    def _raise_alert(self, dist_to_anomaly):
        self.anomalies_found += 1
        wp_x, wp_y = PATROL_ROUTE[self.patrol_idx]

        alert_msg = (
            f"ANOMALY DETECTED | "
            f"Location: WP{self.patrol_idx} ({wp_x:.1f},{wp_y:.1f}) | "
            f"Distance: {dist_to_anomaly:.2f}m | "
            f"Detection #{self.anomalies_found}"
        )
        self.alert_pub.publish(alert_msg)

        rospy.logwarn(
            f"\n{'!'*46}\n"
            f"  ⚠  ANOMALY FOUND — stopping for {ALERT_HOLD_SECS:.0f}s\n"
            f"  Location : WP{self.patrol_idx} ({wp_x:.1f},{wp_y:.1f})\n"
            f"  Distance : {dist_to_anomaly:.2f} m from anomaly\n"
            f"  Total detections: {self.anomalies_found}\n"
            f"  Patrols completed: {self.patrols_completed}\n"
            f"{'!'*46}"
        )

        # Stop robot during alert
        self.vel_pub.publish(Twist())
        self.alert_active = True
        self.alert_start  = rospy.Time.now()

    # ── main loop ──────────────────────────────────────────────────────────
    def _laser_cb(self, msg):
        self._ground_truth()

        # ── alert hold: stay stopped during alert window ───────────────────
        if self.alert_active:
            elapsed = (rospy.Time.now() - self.alert_start).to_sec()
            if elapsed < ALERT_HOLD_SECS:
                self.vel_pub.publish(Twist())   # keep stopped
                return
            else:
                self.alert_active = False
                # Advance to next waypoint immediately so the robot moves
                # away from the anomaly location before detection re-enables.
                self.total_wps_visited += 1
                self.patrol_idx     = (self.patrol_idx + 1) % len(PATROL_ROUTE)
                self.current_target = np.array(PATROL_ROUTE[self.patrol_idx])
                # Block re-detection for 20 s — enough time to clear the area.
                self.detection_cooldown_until = rospy.Time.now() + rospy.Duration(20.0)
                wp_x, wp_y = PATROL_ROUTE[self.patrol_idx]
                rospy.loginfo(
                    f"[DEMO] Alert cleared → moving to WP{self.patrol_idx} "
                    f"({wp_x:.1f},{wp_y:.1f}) | cooldown 20 s"
                )

        dist         = np.linalg.norm(self.current_pos - self.current_target)
        dist_anomaly = np.linalg.norm(self.current_pos - self.anomaly_pos)

        state = self._state(msg, dist)

        # ── anomaly detection (gated by cooldown) ──────────────────────────
        cooldown_active = rospy.Time.now() < self.detection_cooldown_until
        if self.target_visible and dist_anomaly < ANOMALY_DETECT_D and not cooldown_active:
            self._get_anomaly_pos()
            self._raise_alert(dist_anomaly)
            return

        # ── waypoint advancement ───────────────────────────────────────────
        if dist < WP_REACH_DIST:
            self.total_wps_visited += 1
            prev_idx        = self.patrol_idx
            self.patrol_idx = (self.patrol_idx + 1) % len(PATROL_ROUTE)
            self.current_target = np.array(PATROL_ROUTE[self.patrol_idx])

            if self.patrol_idx == 0:
                self.patrols_completed += 1
                rospy.logwarn(
                    f"[DEMO] ✓ Full patrol circuit #{self.patrols_completed} complete | "
                    f"Anomalies found: {self.anomalies_found}"
                )
            else:
                wp_x, wp_y = PATROL_ROUTE[self.patrol_idx]
                rospy.loginfo(
                    f"[DEMO] ✓ WP{prev_idx} cleared → WP{self.patrol_idx} "
                    f"({wp_x:.1f},{wp_y:.1f}) | visited={self.total_wps_visited}"
                )

        # ── stuck recovery ─────────────────────────────────────────────────
        self.pos_history.append(self.current_pos.copy())
        if len(self.pos_history) == 80:
            net = np.linalg.norm(self.pos_history[-1] - self.pos_history[0])
            if net < 0.5:
                self.stuck_count += 1
                self.pos_history.clear()
                spin = Twist()
                spin.angular.z = 0.8 if self.stuck_count % 2 == 1 else -0.8
                self.vel_pub.publish(spin)
                rospy.logwarn(f"[DEMO] Stuck — recovery spin #{self.stuck_count}")
                return

        # ── action from Q-table (pure exploitation) ────────────────────────
        self.q_table.setdefault(state, self._default_q())
        action = self._argmax_tie(self.q_table[state])

        wp_x, wp_y = PATROL_ROUTE[self.patrol_idx]
        rospy.loginfo_throttle(2.0,
            f"[DEMO] WP{self.patrol_idx} ({wp_x:.0f},{wp_y:.0f}) "
            f"dist:{dist:.1f}m | "
            f"anomaly:{dist_anomaly:.1f}m CAM:{'Y🎯' if self.target_visible else 'N'} | "
            f"circuits:{self.patrols_completed} found:{self.anomalies_found}"
        )

        mv = Twist()
        if   action == 0: mv.linear.x  =  0.35
        elif action == 1: mv.angular.z =  0.7
        elif action == 2: mv.angular.z = -0.7
        elif action == 3: mv.linear.x, mv.angular.z =  0.2,  0.45
        else:             mv.linear.x, mv.angular.z =  0.2, -0.45
        self.vel_pub.publish(mv)


if __name__ == '__main__':
    PatrolDemo()
    rospy.spin()
