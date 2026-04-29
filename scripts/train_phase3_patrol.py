#!/usr/bin/env python3
"""
Phase 3 — Full Hospital Patrol with Hidden Anomaly Detection
COM760 CW2 Group 1

GOAL: The robot patrols the full hospital route and finds a hidden anomaly
      using its camera — it does NOT know which waypoint has the anomaly.

HOW:
  - Each episode: anomaly marker spawns at ONE random patrol waypoint (hidden).
  - Robot follows the patrol route sequentially: WP0 → WP1 → ... → WP7 → WP0
  - When the robot reaches a waypoint it advances to the next automatically.
  - SUCCESS: camera detects the green anomaly marker (target_visible = True)
             while within 2.5 m of the anomaly waypoint.
  - This is the realistic patrol scenario — detection by vision, not coordinates.

PREREQUISITE: Run Phase 2 first (~/q_table_p2.pkl must exist).

PATROL_ROUTE: Must match the same waypoints used in Phase 2.
              Verify each coordinate is on open floor in Gazebo.

WHEN TO STOP: SR > 30% is strong for a full patrol task.
              The anomaly could be at the last waypoint (worst case), so some
              episodes will always take a full circuit. That is expected.

OUTPUT: ~/q_table_p3.pkl

RUN:   rosrun com760cw2_group1 train_phase3_patrol.py
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
LOAD_FILE   = os.path.expanduser('~/q_table_p2.pkl')
SAVE_FILE   = os.path.expanduser('~/q_table_p3.pkl')

# Derived from furniture positions in hospital.world (must match Phase 2 exactly).
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

WP_REACH_DIST    = 1.5   # distance to advance to next waypoint
ANOMALY_DETECT_D = 2.5   # camera detection range (must be in this radius)
MAX_STEPS        = 3500  # full patrol circuit is long — allow more steps


class Phase3Agent:
    def __init__(self):
        rospy.init_node('patrol_phase3_node')

        self.episode_count   = 0
        self.success_count   = 0
        self.collision_count = 0
        self.step_count      = 0
        self.total_ep_reward = 0.0
        self.waypoints_visited = 0   # count per episode

        self.spawn_grace  = 0
        self.pos_history  = deque(maxlen=80)
        self.stuck_count  = 0

        self.last_dist   = None
        self.last_state  = None
        self.last_action = None

        self.current_pos = np.array([0.0, 0.0])
        self.prev_pos    = np.array([0.0, 0.0])
        self.robot_yaw   = 0.0

        # Navigation target = current patrol waypoint (NOT the anomaly)
        self.patrol_idx     = 0
        self.current_target = np.array(PATROL_ROUTE[0])

        # Anomaly position is hidden — set each episode, not used for navigation
        self.anomaly_pos    = None
        self.anomaly_wp_idx = -1

        self.alpha         = 0.1    # lower lr — mostly exploiting P2 knowledge
        self.gamma         = 0.95
        self.epsilon       = 0.3    # start mostly exploiting
        self.epsilon_min   = 0.02
        self.epsilon_decay = 0.999  # very slow — long episodes

        self.goal_threshold = WP_REACH_DIST

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

        self._start_episode()
        rospy.loginfo(
            "\n╔════════════════════════════════════════════════╗\n"
            "║  PHASE 3 — FULL PATROL + ANOMALY DETECTION    ║\n"
            "║  Robot patrols all rooms, finds anomaly        ║\n"
            "║  by camera — it does NOT know which waypoint  ║\n"
            f"║  {len(PATROL_ROUTE)} waypoints | EP {self.episode_count:04d} | EPS {self.epsilon:.3f}  ║\n"
            "╚════════════════════════════════════════════════╝"
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
        # dist here is distance to current patrol waypoint (nav target)
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

        ang = math.atan2(self.current_target[1] - self.current_pos[1],
                         self.current_target[0] - self.current_pos[0])
        err = ang - self.robot_yaw
        while err >  math.pi: err -= 2*math.pi
        while err < -math.pi: err += 2*math.pi
        hd = int(((err + math.pi) / (2*math.pi)) * 4) % 4

        # State: (left, center, right, heading) → 3^3 × 4 = 108 states
        # Distance and camera removed from state (camera used for reward only)
        return str(ls + (hd,))

    # ── episode start ──────────────────────────────────────────────────────
    def _start_episode(self):
        self._ground_truth()
        self.spawn_grace = 20

        # Place anomaly at a random patrol waypoint (unknown to navigation)
        self.anomaly_wp_idx = random.randint(0, len(PATROL_ROUTE) - 1)
        ax, ay = PATROL_ROUTE[self.anomaly_wp_idx]
        self.anomaly_pos = np.array([ax, ay])

        # Bot always starts patrolling from waypoint 0
        self.patrol_idx     = 0
        self.current_target = np.array(PATROL_ROUTE[self.patrol_idx])
        self.waypoints_visited = 0

        # Place the marker at the hidden anomaly waypoint
        s = ModelState()
        s.model_name      = 'target_marker'
        s.pose.position.x = ax
        s.pose.position.y = ay
        s.pose.position.z = 0.05
        try:
            self.set_state_proxy(s)
        except Exception:
            pass

        rospy.loginfo(
            f"[P3] Episode {self.episode_count} | "
            f"Anomaly hidden at WP{self.anomaly_wp_idx} "
            f"({ax:.1f},{ay:.1f}) — robot does not know this"
        )

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
        self._start_episode()

    # ── Q-table ────────────────────────────────────────────────────────────
    def _load(self):
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
                        self.epsilon       = d.get('epsilon',   0.3)
                        self.episode_count = d.get('episodes',  0)
                        self.success_count = d.get('successes', 0)
                    rospy.loginfo(f"[P3] Loaded {len(self.q_table)} Q-states from {path}")
                    return
                except Exception as e:
                    rospy.logwarn(f"[P3] Load failed ({path}): {e}")
        rospy.logwarn("[P3] No Q-table found. Run Phases 1 and 2 first.")

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

        # dist = distance to current PATROL WAYPOINT (for navigation)
        dist = np.linalg.norm(self.current_pos - self.current_target)
        # dist_anomaly = distance to hidden anomaly (for detection only)
        dist_anomaly = np.linalg.norm(self.current_pos - self.anomaly_pos)

        if self.last_dist is None:
            self.last_dist = dist
        progress = self.last_dist - dist

        state = self._state(msg, dist)

        collision = self._safe_min(msg.ranges[216:504]) < 0.45

        self.pos_history.append(self.current_pos.copy())
        stuck = (len(self.pos_history) == 80 and
                 np.linalg.norm(self.pos_history[-1] - self.pos_history[0]) < 0.5)

        # ── anomaly detected? ─────────────────────────────────────────────
        anomaly_found = self.target_visible and dist_anomaly < ANOMALY_DETECT_D

        # ── reached current patrol waypoint? → advance to next ────────────
        if dist < WP_REACH_DIST and not anomaly_found:
            self.waypoints_visited += 1
            self.patrol_idx = (self.patrol_idx + 1) % len(PATROL_ROUTE)
            self.current_target = np.array(PATROL_ROUTE[self.patrol_idx])
            wp_x, wp_y = PATROL_ROUTE[self.patrol_idx]
            rospy.loginfo(
                f"[P3] ✓ WP reached → next: WP{self.patrol_idx} "
                f"({wp_x:.1f},{wp_y:.1f}) | visited={self.waypoints_visited}"
            )

        # ── heading angle error (to current patrol waypoint) ────────────────
        front_laser = self._safe_min(msg.ranges[240:480])
        angle_to_target = math.atan2(self.current_target[1] - self.current_pos[1],
                                     self.current_target[0] - self.current_pos[0])
        err = angle_to_target - self.robot_yaw
        while err > math.pi: err -= 2 * math.pi
        while err < -math.pi: err += 2 * math.pi

        # ── reward ────────────────────────────────────────────────────────
        reward      = 0.0
        is_terminal = False

        # Proximity penalty: extra penalty for being too close to obstacles
        if front_laser < 1.5:
            reward -= (1.5 - front_laser) **2 * 500.0

        if collision and self.spawn_grace <= 0:
            reward, is_terminal = -5000.0, True
        elif anomaly_found:
            # Bonus for finding early (fewer waypoints visited = faster patrol)
            efficiency_bonus = max(0, len(PATROL_ROUTE) - self.waypoints_visited) * 30
            reward      = 10000.0 + efficiency_bonus
            is_terminal = True
        elif stuck and self.spawn_grace <= 0:
            self.stuck_count += 1
            self.pos_history.clear()
            penalty = min(30.0 * self.stuck_count, 150.0)
            reward = -penalty
            spin = Twist()
            spin.linear.x = -0.08
            spin.angular.z = 0.7 if self.stuck_count % 2 == 1 else -0.7
            self.vel_pub.publish(spin)
            rospy.logwarn(f"[P3] STUCK #{self.stuck_count} — recovery turn/back (penalty={penalty:.0f})")
            self.last_state = self.last_action = self.last_dist = None
            return
        elif self.step_count > 2500:
            reward, is_terminal = -100.0, True
        else:
            # Progress reward: moving closer to current waypoint
            reward += progress * 500.0
            # Light penalty for moving away from target
            if progress < 0:
                reward -= 50.0
            
            front = self._safe_min(msg.ranges[216:504])
            if front < 1.2:
                reward -= (1.2 - front) * 5.0
            
            # Camera rewards: seeing the anomaly is a strong signal
            if self.target_visible:
                reward += 2.0
                if abs(self.visual_error) < 0.25:
                    reward += 1.0

            # Strong heading-alignment bonus: reward when pointing toward target
            heading_bonus = max(0.0, 1.0 - abs(err) / math.pi)
            reward += heading_bonus * 8.0
            
            # Mild penalty for turning away: if heading error is large, penalize
            if abs(err) > math.radians(90):
                reward -= 10.0
            elif abs(err) > math.radians(60):
                reward -= 5.0
            
            reward -= 0.1 * dist

        self.total_ep_reward += reward
        self.step_count      += 1

        if self.last_state is not None and self.last_action is not None:
            self.q_table.setdefault(self.last_state, self._default_q())
            self.q_table.setdefault(state, self._default_q())
            mx = max(self.q_table[state])
            self.q_table[self.last_state][self.last_action] += self.alpha * (
                reward + self.gamma * mx - self.q_table[self.last_state][self.last_action]
            )

        if is_terminal:
            if anomaly_found:
                reason = f"ANOMALY FOUND (WP{self.anomaly_wp_idx})"
                self.success_count += 1
            elif collision:
                reason = "COLLISION"
                self.collision_count += 1
            else:
                reason = f"TIMEOUT (anomaly was at WP{self.anomaly_wp_idx})"

            sr = self.success_count / max(self.episode_count + 1, 1) * 100
            rospy.logwarn(
                f"\n{'='*46}\n"
                f" [P3] EP {self.episode_count:04d} → {reason}\n"
                f" Steps:{self.step_count} | WPs visited:{self.waypoints_visited}\n"
                f" Reward:{self.total_ep_reward:.0f} | Dist to anomaly:{dist_anomaly:.1f}m\n"
                f" SR:{sr:.1f}% ({self.success_count}/{self.episode_count+1})\n"
                f" EPS:{self.epsilon:.4f} | Q-states:{len(self.q_table)}\n"
                f"{'='*46}"
            )

            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            self.episode_count += 1
            self._save()
            self._reset()
            self.last_state = self.last_dist = None
            self.pos_history.clear()
            self.stuck_count     = 0
            self.total_ep_reward = self.step_count = 0
            self.waypoints_visited = 0
            return

        # Q-learning action selection with reactive obstacle override
        left_dist = self._safe_min(msg.ranges[0:240])
        right_dist = self._safe_min(msg.ranges[480:720])

        self.q_table.setdefault(state, self._default_q())
        if front_laser < 0.9:
            # Obstacle ahead: force a turn toward the more open side
            if left_dist > right_dist:
                action = 1
            else:
                action = 2
        else:
            # Bias toward forward when well-aligned with target
            if abs(err) < math.radians(30):
                if random.random() < 0.7:
                    action = 0
                else:
                    action = (random.choice(self.actions) if random.random() < self.epsilon
                              else self._argmax_tie(self.q_table[state]))
            else:
                action = (random.choice(self.actions) if random.random() < self.epsilon
                          else self._argmax_tie(self.q_table[state]))

        mv = Twist()
        if action == 0:
            mv.linear.x = 0.35
        elif action == 1:
            mv.angular.z = 0.7
            if front_laser < 0.9:
                mv.linear.x = 0.06
        elif action == 2:
            mv.angular.z = -0.7
            if front_laser < 0.9:
                mv.linear.x = 0.06
        elif action == 3:
            mv.linear.x, mv.angular.z = 0.2, 0.45
        else:
            mv.linear.x, mv.angular.z = 0.2, -0.45

        if front_laser > 1.2 and mv.linear.x > 0.0:
            reward += mv.linear.x * 20.0  # strong bonus for forward motion on a clear path
            # Extra bonus for forward motion when well-aligned with target
            if abs(err) < math.radians(45):
                reward += mv.linear.x * 15.0

        self.total_ep_reward += reward
        self.step_count      += 1

        names = {0:"FWD",1:"LEFT",2:"RIGHT",3:"FWD+L",4:"FWD+R"}
        sr = self.success_count / max(self.episode_count, 1) * 100
        rospy.loginfo_throttle(1.5,
            f"[P3 EP{self.episode_count:04d}|{self.step_count:04d}] "
            f"HEADING:{math.degrees(err):.1f}° NAV→WP{self.patrol_idx}:{dist:.1f}m "
            f"ANOMALY:{dist_anomaly:.1f}m CAM:{'Y' if self.target_visible else 'N'} SR:{sr:.1f}%")

        self.vel_pub.publish(mv)

        self.last_state  = state
        self.last_action = action
        self.last_dist   = dist


if __name__ == '__main__':
    Phase3Agent()
    rospy.spin()
