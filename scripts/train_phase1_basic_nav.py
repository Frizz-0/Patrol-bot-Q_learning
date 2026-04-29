#!/usr/bin/env python3
"""
Phase 1 — Basic Navigation Training
COM760 CW2 Group 1

GOAL: Teach the robot to avoid walls and reach a nearby target.

HOW:  Target spawns at a random angle, 1.5–5 m from the robot (curriculum).
      Distance grows slowly as the agent succeeds, so early episodes are
      easy enough to generate a success signal.

WHEN TO STOP: Run until the terminal shows SR (success rate) above ~60–70%.
              Typically 300–500 episodes (~4–8 hours real time).

OUTPUT: ~/q_table_p1.pkl   (loaded by Phase 2)

RUN:   rosrun com760cw2_group1 train_phase1_basic_nav.py
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
SAVE_FILE   = os.path.expanduser('~/q_table_p1.pkl')


class Phase1Agent:
    def __init__(self):
        rospy.init_node('patrol_phase1_node')

        self.episode_count   = 0
        self.success_count   = 0
        self.collision_count = 0
        self.step_count      = 0
        self.total_ep_reward = 0.0

        self.spawn_grace     = 0
        self.pos_history     = deque(maxlen=80)  # 4 s window at 20 Hz
        self.stuck_count     = 0   # stuck events this episode; escalating penalty, never terminal

        self.last_dist   = None
        self.last_state  = None
        self.last_action = None

        self.current_pos = np.array([0.0, 0.0])
        self.prev_pos    = np.array([0.0, 0.0])
        self.robot_yaw   = 0.0

        self.alpha         = 0.2
        self.gamma         = 0.95
        self.epsilon       = 1.0
        self.epsilon_min   = 0.05
        self.epsilon_decay = 0.997

        self.current_target = np.array([0.0, 0.0])
        self.goal_threshold = 1.5

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
            "\n╔══════════════════════════════════════╗\n"
            "║  PHASE 1 — BASIC NAVIGATION          ║\n"
            "║  Goal : reach random nearby target   ║\n"
            f"║  EP {self.episode_count:04d} | EPS {self.epsilon:.3f}           ║\n"
            "╚══════════════════════════════════════╝"
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
        # 3 laser sectors: left, center, right
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
        # Distance and camera removed from state (camera used for rewards only)
        return str(ls + (hd,))

    # ── respawn ────────────────────────────────────────────────────────────
    def _respawn(self):
        self._ground_truth()
        self.spawn_grace = 15

        # PHASE 1: Always 2m in front, small random variation (±20°)
        # This teaches the agent that heading-0 + forward = success
        angle_offset = random.uniform(-0.35, 0.35)  # ±20° around front
        d = 2.0
        x = self.current_pos[0] + d * math.cos(angle_offset)
        y = self.current_pos[1] - d * math.sin(angle_offset)

        self.current_target = np.array([x, y])
        s = ModelState()
        s.model_name      = 'target_marker'
        s.pose.position.x = x
        s.pose.position.y = y
        s.pose.position.z = 0.05
        try:
            self.set_state_proxy(s)
        except Exception:
            pass
        rospy.loginfo(f"[P1] Target ({x:.2f},{y:.2f}) | 2m front, ±20° variation")

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
        if not os.path.exists(SAVE_FILE):
            return
        try:
            d = pickle.load(open(SAVE_FILE, 'rb'))
            raw = d.get('q_table', {})
            self.q_table = {
                k: (v + [0.0]*(NUM_ACTIONS-len(v)))[:NUM_ACTIONS]
                for k, v in raw.items()
            }
            self.epsilon       = d.get('epsilon',   1.0)
            self.episode_count = d.get('episodes',  0)
            self.success_count = d.get('successes', 0)
            rospy.loginfo(f"[P1] Loaded {len(self.q_table)} states | EP {self.episode_count}")
        except Exception as e:
            rospy.logwarn(f"[P1] Load failed: {e}")

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
        # Stuck = less than 0.5 m net displacement over 4 s (80 steps).
        # On first/second detection: apply recovery spin + penalty, don't end episode.
        # On third detection: give up — robot is truly trapped.
        stuck = (len(self.pos_history) == 80 and
                 np.linalg.norm(self.pos_history[-1] - self.pos_history[0]) < 0.5)

        reward      = -0.3
        is_terminal = False

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
            rospy.logwarn(f"[P1] STUCK #{self.stuck_count} — recovery spin (penalty={penalty:.0f})")
            self.last_state = self.last_action = self.last_dist = None
            return
        elif dist < self.goal_threshold:
            reward = 10000.0 + max(0, (1500 - self.step_count)) * 0.3
            is_terminal = True
        elif self.step_count > 1500:
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
                      "COLLISION" if collision else "TIMEOUT")
            if reason == "SUCCESS":
                self.success_count += 1
            sr = self.success_count / max(self.episode_count + 1, 1) * 100
            rospy.logwarn(
                f"\n{'='*40}\n"
                f" [P1] EP {self.episode_count:04d} → {reason}\n"
                f" Steps:{self.step_count} Reward:{self.total_ep_reward:.0f}\n"
                f" SR:{sr:.1f}% ({self.success_count}/{self.episode_count+1})\n"
                f" EPS:{self.epsilon:.4f} Q-states:{len(self.q_table)}\n"
                f"{'='*40}"
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
            f"[P1 EP{self.episode_count:04d}|{self.step_count:04d}] "
            f"{names[action]} DIST:{dist:.2f} R:{reward:.1f} "
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
    Phase1Agent()
    rospy.spin()
