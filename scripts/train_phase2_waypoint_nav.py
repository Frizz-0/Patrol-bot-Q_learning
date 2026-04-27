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
PATROL_ROUTE = [
    ( 0.0, 10.0),   # spawn / home corridor
    ( 3.0, 10.5),   # east corridor
    ( 6.0, 10.0),   # east wing entrance
    ( 5.5,  7.5),   # south-east room
    ( 2.5,  7.0),   # central area
    ( 0.0,  7.5),   # south corridor
    (-3.0,  7.5),   # west room
    (-3.0, 10.5),   # west corridor
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

        self.alpha         = 0.15   # slightly lower — fine-tuning on top of P1
        self.gamma         = 0.95
        self.epsilon       = 0.5    # start mid-range since P1 gave us a base
        self.epsilon_min   = 0.05
        self.epsilon_decay = 0.998  # decay slower — longer episodes

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

    # ── respawn ────────────────────────────────────────────────────────────
    def _respawn(self):
        self._ground_truth()
        self.spawn_grace = 20   # longer grace — farther targets take more time to orient

        # Pick a random patrol waypoint that's at least 2 m from the robot
        wp = random.choice(PATROL_ROUTE)
        for _ in range(15):
            wp = random.choice(PATROL_ROUTE)
            if np.linalg.norm(self.current_pos - np.array(wp)) > 2.0:
                break

        x, y = wp
        self.current_target = np.array([x, y])
        target_dist = np.linalg.norm(self.current_pos - self.current_target)

        s = ModelState()
        s.model_name      = 'target_marker'
        s.pose.position.x = x
        s.pose.position.y = y
        s.pose.position.z = 0.05
        try:
            self.set_state_proxy(s)
        except Exception:
            pass
        rospy.loginfo(f"[P2] Waypoint ({x:.1f},{y:.1f}) | dist={target_dist:.1f}m")

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

        reward      = -0.3
        is_terminal = False

        if collision and self.spawn_grace <= 0:
            reward, is_terminal = -200.0, True
        elif stuck and self.spawn_grace <= 0:
            self.stuck_count += 1
            self.pos_history.clear()
            if self.stuck_count >= 3:
                reward, is_terminal = -100.0, True
            else:
                reward = -30.0
                spin = Twist()
                spin.angular.z = 0.8 if self.stuck_count % 2 == 1 else -0.8
                self.vel_pub.publish(spin)
                rospy.logwarn(f"[P2] STUCK #{self.stuck_count} — recovery spin")
                self.last_state = self.last_action = self.last_dist = None
                return
        elif dist < self.goal_threshold:
            # Longer episodes → scale time bonus accordingly
            reward = 500.0 + max(0, (2000 - self.step_count)) * 0.2
            is_terminal = True
        elif self.step_count > 2000:
            reward, is_terminal = -50.0, True
        else:
            reward += progress * 300.0
            front = self._safe_min(msg.ranges[216:504])
            if front < 1.2:
                reward -= (1.2 - front) * 5.0
            if self.target_visible:
                reward += 2.0
                if abs(self.visual_error) < 0.25:
                    reward += 1.0
            reward -= 0.02 * dist

        self.total_ep_reward += reward
        self.step_count      += 1

        if self.last_state is not None:
            self.q_table.setdefault(self.last_state, self._default_q())
            self.q_table.setdefault(state,           self._default_q())
            mx = max(self.q_table[state])
            self.q_table[self.last_state][self.last_action] += self.alpha * (
                reward + self.gamma * mx - self.q_table[self.last_state][self.last_action]
            )

        if is_terminal:
            reason = ("SUCCESS"   if dist < self.goal_threshold else
                      "COLLISION" if collision else
                      "STUCK"     if stuck else "TIMEOUT")
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

        self.q_table.setdefault(state, self._default_q())
        action = (random.choice(self.actions) if random.random() < self.epsilon
                  else self._argmax_tie(self.q_table[state]))

        names = {0:"FWD",1:"LEFT",2:"RIGHT",3:"FWD+L",4:"FWD+R"}
        sr = self.success_count / max(self.episode_count, 1) * 100
        rospy.loginfo_throttle(1.0,
            f"[P2 EP{self.episode_count:04d}|{self.step_count:04d}] "
            f"{names[action]} DIST:{dist:.2f}m R:{reward:.1f} "
            f"CAM:{'Y' if self.target_visible else 'N'} SR:{sr:.1f}%")

        mv = Twist()
        if   action == 0: mv.linear.x  =  0.35
        elif action == 1: mv.angular.z =  0.7
        elif action == 2: mv.angular.z = -0.7
        elif action == 3: mv.linear.x, mv.angular.z =  0.2,  0.45
        else:             mv.linear.x, mv.angular.z =  0.2, -0.45
        self.vel_pub.publish(mv)

        self.last_state  = state
        self.last_action = action
        self.last_dist   = dist


if __name__ == '__main__':
    Phase2Agent()
    rospy.spin()
