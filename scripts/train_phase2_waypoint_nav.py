#!/usr/bin/env python3
"""
Phase 2 — Hospital Waypoint Navigation Training
COM760 CW2 Group 1

GOAL: Teach the robot to navigate to specific locations across the hospital.

HOW:  Each episode picks a random waypoint from the patrol route as the target.
      The robot must navigate there from the spawn point (0, 10).
      This forces the agent to learn long-range navigation across the full
      hospital, not just reaching nearby random points.

PREREQUISITE: Run Phase 1 first (~/q_table_p1.pkl must exist).

PATROL_ROUTE: Open Gazebo, use the mouse to hover over floor positions and
              read the (x, y) coordinates from the status bar. Replace the
              values below with open corridor/room positions in YOUR world.
              All points must be on navigable floor, not inside walls.

WHEN TO STOP: SR > 40% sustained over 50+ episodes, typically 400–600 episodes.

OUTPUT: ~/q_table_p2.pkl   (loaded by Phase 3)

RUN:   rosrun com760cw2_group1 train_phase2_waypoint_nav.py
"""
import os, rospy, numpy as np, random, math, pickle
from collections import deque
from sensor_msgs.msg import LaserScan, Image
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState, GetModelState
from tf.transformations import euler_from_quaternion
from cv_bridge import CvBridge
import cv2

NUM_ACTIONS = 5
LOAD_FILE   = os.path.expanduser('~/q_table_p1.pkl')
SAVE_FILE   = os.path.expanduser('~/q_table_p2.pkl')

# ── PATROL WAYPOINTS ───────────────────────────────────────────────────────
# IMPORTANT: Verify these in Gazebo before running.
# Robot spawns at (0, 10). Hover over the floor in Gazebo to get coordinates.
# Each point should be in an open corridor or room, not inside a wall.
# Waypoints derived from furniture positions in hospital.world.
# Furniture can only sit on open floor, so these zones are confirmed navigable.
PATROL_ROUTE = [
    ( 0.0, 10.0),   # WP0 - spawn / north main corridor
    (-5.0,  7.0),   # WP1 - west waiting area  (chairs at -4.8,6.5 & -7.9,6.5)
    ( 0.0,  3.0),   # WP2 - front lobby         (nurses station at 0,1.5)
    ( 5.0,  7.0),   # WP3 - east waiting area   (chairs at 5.2,6.6 & 8.2,6.6)
    ( 0.0, 16.0),   # WP4 - north corridor       (between spawn & elevators at y=19.5)
    ( 0.0, -4.0),   # WP5 - south junction       (table confirmed at 1.2,-5.6)
    (-8.0,-12.0),   # WP6 - south-west wing      (curtains/beds at x≈-11,y≈-14)
    ( 8.0,-17.0),   # WP7 - south-east wing      (curtains at 11.1,-17.8 to -21.4)
]


class Phase2Agent:
    def __init__(self):
        rospy.init_node('patrol_phase2_node')

        self.episode_count   = 0
        self.success_count   = 0
        self.collision_count = 0
        self.step_count      = 0
        self.total_ep_reward = 0.0

        self.spawn_grace  = 0
        self.pos_history  = deque(maxlen=80)
        self.stuck_count  = 0

        self.last_dist   = None
        self.last_state  = None
        self.last_action = None

        self.current_pos = np.array([0.0, 0.0])
        self.prev_pos    = np.array([0.0, 0.0])
        self.robot_yaw   = 0.0

        self.alpha         = 0.25   # slightly lower — fine-tuning on top of P1
        self.gamma         = 0.95
        self.epsilon       = 0.8    # start mid-range since P1 gave us a base
        self.epsilon_min   = 0.05
        self.epsilon_decay = 0.993  # decay slower — longer episodes

        self.current_target = np.array([0.0, 0.0])
        self.goal_threshold = 2.0   # slightly wider for longer-range targets

        self.bridge         = CvBridge()
        self.target_visible = False
        self.visual_error   = 0.0

        self.actions = list(range(NUM_ACTIONS))
        self.q_table = {}
        self._load()

        self.reset_proxy     = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self.set_state_proxy = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.get_model_proxy = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)

        self.vel_pub = rospy.Publisher('/Group1Bot/cmd_vel', Twist, queue_size=10)
        rospy.Subscriber('/Group1Bot/camera/image_raw', Image, self._image_cb)
        rospy.Subscriber('/Group1Bot/laser/scan', LaserScan, self._laser_cb)

        self._respawn()
        rospy.loginfo(
            "\n╔══════════════════════════════════════════╗\n"
            "║  PHASE 2 — HOSPITAL WAYPOINT NAVIGATION  ║\n"
            "║  Goal : reach patrol waypoints by name   ║\n"
            f"║  {len(PATROL_ROUTE)} waypoints loaded | EP {self.episode_count:04d}       ║\n"
            "╚══════════════════════════════════════════╝"
        )

    # ── camera ────────────────────────────────────────────────────────────
    def _image_cb(self, msg):
        try:
            hsv = cv2.cvtColor(self.bridge.imgmsg_to_cv2(msg, 'bgr8'),
                               cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, np.array([40, 80, 80]), np.array([80, 255, 255]))
            M = cv2.moments(mask)
            if M['m00'] > 300:
                self.visual_error   = (int(M['m10']/M['m00']) - 320) / 320.0
                self.target_visible = True
            else:
                self.target_visible = False
                self.visual_error   = 0.0
        except Exception:
            pass

    # ── utils ─────────────────────────────────────────────────────────────
    def _safe_min(self, ranges):
        v = [x for x in ranges if math.isfinite(x) and x > 0.01]
        return min(v) if v else float('inf')

    # For states NOT in Phase 1 Q-table, learn faster
    def compute_alpha(self, state):
        if state in self.q_table:
            return 0.15  # trust Phase 1, learn slowly
        else:
            return 0.3   # new state, learn aggressively
    
    def _ground_truth(self):
        try:
            res = self.get_model_proxy('Group1Bot', 'world')
            self.prev_pos    = self.current_pos.copy()
            self.current_pos = np.array([res.pose.position.x, res.pose.position.y])
            q = res.pose.orientation
            _, _, self.robot_yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        except Exception:
            pass

    def _argmax_tie(self, q):
        m = max(q)
        return random.choice([i for i, v in enumerate(q) if v == m])

    def _default_q(self):
        return [0.0] * NUM_ACTIONS

    # ── state ─────────────────────────────────────────────────────────────
    def _state(self, msg, dist):
        # 3 laser sectors: left, center, right (NOT distance-dependent)
        def level(r):
            d = self._safe_min(r)
            return 0 if d < 0.5 else (1 if d < 1.2 else 2)
        
        # Left: idx 0-240 (-90° to -30°)
        # Center: idx 240-480 (-30° to +30°)
        # Right: idx 480-720 (+30° to +90°)
        ls = (
            level(msg.ranges[0:240]),      # left
            level(msg.ranges[240:480]),    # center (front)
            level(msg.ranges[480:720])     # right
        )

        # 4 heading bins (instead of 8) to simplify target direction encoding
        ang = math.atan2(self.current_target[1] - self.current_pos[1],
                         self.current_target[0] - self.current_pos[0])
        err = ang - self.robot_yaw
        while err >  math.pi: err -= 2*math.pi
        while err < -math.pi: err += 2*math.pi
        hd = int(((err + math.pi) / (2*math.pi)) * 4) % 4

        # State: (left, center, right, heading) → 3^3 × 4 = 108 states
        # Camera is used for reward shaping only, NOT in state
        return str(ls + (hd,))

    # ── respawn ────────────────────────────────────────────────────────────
    def _respawn(self):
        self._ground_truth()
        self.spawn_grace = 20

        # Leash logic: target directly in front, distance increases with episodes
        # Start at 2m, increase by 0.05m per episode, max 15m
        target_dist = min(2.0 + self.episode_count * 0.05, 15.0)
        
        angle = random.uniform(0, 2*math.pi)  # randomize target angle for more diverse navigation
        # Target in front of current position (robot faces +y at spawn)
        # target_x = self.current_pos[0] + target_dist * math.sin(self.robot_yaw)
        # target_y = self.current_pos[1] + target_dist * math.cos(self.robot_yaw)
        target_x = self.current_pos[0] + target_dist * math.cos(angle)
        target_y = self.current_pos[1] + target_dist * math.sin(angle)

        self.current_target = np.array([target_x, target_y])

        s = ModelState()
        s.model_name      = 'target_marker'
        s.pose.position.x = target_x
        s.pose.position.y = target_y
        s.pose.position.z = 0.05
        try:
            self.set_state_proxy(s)
        except Exception:
            pass
        rospy.loginfo(f"[P2] Leash target ({target_x:.2f},{target_y:.2f}) dist={target_dist:.1f}m | EP {self.episode_count}")

    # ── reset ──────────────────────────────────────────────────────────────
    def _reset(self):
        self.vel_pub.publish(Twist())
        r = ModelState()
        r.model_name         = 'Group1Bot'
        r.pose.position.x    = 0.0
        r.pose.position.y    = 10.0
        r.pose.orientation.w = 1.0
        try:
            self.set_state_proxy(r)
        except Exception:
            self.reset_proxy()
        rospy.sleep(0.5)
        self._respawn()

    # ── Q-table ────────────────────────────────────────────────────────────
    def _load(self):
        # Try Phase 2 save first (resume), then fall back to Phase 1
        for path in [SAVE_FILE, LOAD_FILE]:
            if os.path.exists(path):
                try:
                    d = pickle.load(open(path, 'rb'))
                    raw = d.get('q_table', {})
                    self.q_table = {
                        k: (v + [0.0]*(NUM_ACTIONS-len(v)))[:NUM_ACTIONS]
                        for k, v in raw.items()
                    }
                    if path == SAVE_FILE:
                        self.epsilon       = d.get('epsilon',   0.5)
                        self.episode_count = d.get('episodes',  0)
                        self.success_count = d.get('successes', 0)
                    rospy.loginfo(f"[P2] Loaded {len(self.q_table)} Q-states from {path}")
                    return
                except Exception as e:
                    rospy.logwarn(f"[P2] Load failed ({path}): {e}")
        rospy.logwarn("[P2] No Q-table found. Run Phase 1 first for best results.")

    def _save(self):
        pickle.dump({'q_table':   self.q_table,
                     'epsilon':   self.epsilon,
                     'episodes':  self.episode_count,
                     'successes': self.success_count},
                    open(SAVE_FILE, 'wb'))

    # ── main loop ─────────────────────────────────────────────────────────
    def _laser_cb(self, msg):
        if self.spawn_grace > 0:
            self.spawn_grace -= 1

        self._ground_truth()
        dist = np.linalg.norm(self.current_pos - self.current_target)
        if self.last_dist is None:
            self.last_dist = dist
        progress = self.last_dist - dist

        state = self._state(msg, dist)

        collision = self._safe_min(msg.ranges[216:504]) < 0.45

        self.pos_history.append(self.current_pos.copy())
        stuck = (len(self.pos_history) == 80 and
                 np.linalg.norm(self.pos_history[-1] - self.pos_history[0]) < 0.5)

        reward      = 0.0
        is_terminal = False

        front_laser = self._safe_min(msg.ranges[240:480])
        if front_laser < 1.5:
            reward -= (1.5 - front_laser) **2 * 500.0  # heavy penalty for being too close to obstacles
        
        if collision and self.spawn_grace <= 0:
            reward, is_terminal = -5000.0, True
        elif stuck and self.spawn_grace <= 0:
            self.stuck_count += 1
            self.pos_history.clear()
            penalty = min(30.0 * self.stuck_count, 150.0)
            reward = -penalty
            spin = Twist()
            spin.angular.z = 0.8 if self.stuck_count % 2 == 1 else -0.8
            self.vel_pub.publish(spin)
            rospy.logwarn(f"[P2] STUCK #{self.stuck_count} — recovery spin (penalty={penalty:.0f})")
            self.last_state = self.last_action = self.last_dist = None
            return
        elif dist < self.goal_threshold:
            # Longer episodes → scale time bonus accordingly
            reward = 10000.0 + max(0, (2000 - self.step_count)) * 0.2
            is_terminal = True
        elif self.step_count > 2000:
            reward, is_terminal = -100.0, True
        else:
            reward += progress * 500.0
            front = self._safe_min(msg.ranges[216:504])
            if front < 1.2:
                reward -= (1.2 - front) * 5.0
            # if front > 1.5:
            #     reward += 2.0
            if self.target_visible:
                reward += 2.0
                if abs(self.visual_error) < 0.25:
                    reward += 1.0
            reward -= 0.1 * dist

        # Direct steering from ground truth instead of table actions
        angle_to_target = math.atan2(self.current_target[1] - self.current_pos[1],
                                     self.current_target[0] - self.current_pos[0])
        err = angle_to_target - self.robot_yaw
        while err > math.pi: err -= 2 * math.pi
        while err < -math.pi: err += 2 * math.pi

        mv = Twist()
        mv.angular.z = max(-1.0, min(1.0, 1.0 * err))

        if front_laser > 1.5:
            forward_factor = 1.0
        elif front_laser > 1.2:
            forward_factor = 0.4
        else:
            forward_factor = 0.0

        orientation_factor = max(0.0, 1.0 - abs(err) / (math.pi / 2))
        mv.linear.x = 0.35 * forward_factor * orientation_factor

        if front_laser > 1.5:
            reward += mv.linear.x * 20.0  # heavy bonus for forward movement

        self.total_ep_reward += reward
        self.step_count      += 1

        if is_terminal:
            reason = ("SUCCESS"   if dist < self.goal_threshold else
                      "COLLISION" if collision else "TIMEOUT")
            if reason == "SUCCESS":
                self.success_count += 1
            sr = self.success_count / max(self.episode_count + 1, 1) * 100
            rospy.logwarn(
                f"\n{'='*42}\n"
                f" [P2] EP {self.episode_count:04d} → {reason}\n"
                f" Steps:{self.step_count} Reward:{self.total_ep_reward:.0f}\n"
                f" Target:({self.current_target[0]:.1f},{self.current_target[1]:.1f})\n"
                f" SR:{sr:.1f}% EPS:{self.epsilon:.4f} Q:{len(self.q_table)}\n"
                f"{'='*42}"
            )
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            self.episode_count += 1
            self._save()
            self._reset()
            self.last_state = self.last_dist = None
            self.pos_history.clear()
            self.stuck_count     = 0
            self.total_ep_reward = self.step_count = 0
            return

        sr = self.success_count / max(self.episode_count, 1) * 100
        rospy.loginfo_throttle(1.0,
            f"[P2 EP{self.episode_count:04d}|{self.step_count:04d}] "
            f"PCTRL ERR:{math.degrees(err):.1f}deg DIST:{dist:.2f}m R:{reward:.1f} "
            f"CAM:{'Y' if self.target_visible else 'N'} SR:{sr:.1f}%")

        self.vel_pub.publish(mv)

        self.last_dist = dist


if __name__ == '__main__':
    Phase2Agent()
    rospy.spin()
