#!/usr/bin/env python3
"""
Hospital Patrol Bot - Q-Learning Agent
COM760 CW2 Group 1

State space  : laser(243) × heading(8) × dist_bin(8) × camera(4) = 62208 states
Actions      : 5  (forward, left, right, fwd+left curve, fwd+right curve)
Goal         : navigate to anomaly target marker in hospital environment
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


class QLearningAgent:
    def __init__(self):
        rospy.init_node('q_learning_agent_node')

        # ── episode state ──────────────────────────────────────────────────
        self.episode_count      = 0
        self.total_ep_reward    = 0.0
        self.step_count         = 0
        self.success_count      = 0
        self.collision_count    = 0

        self.spawn_grace_steps  = 15
        self.current_grace      = 0
        # Rolling window: track positions over last 50 steps (2.5 s at 20 Hz).
        # Net displacement < 0.3 m over that window = genuinely stuck.
        # Single-step check incorrectly flags turning-in-place as stuck.
        self.pos_history        = deque(maxlen=50)

        self.last_dist          = None
        self.last_state         = None
        self.last_action        = None

        self.current_pos        = np.array([0.0, 0.0])
        self.prev_pos           = np.array([0.0, 0.0])
        self.robot_yaw          = 0.0

        # ── hyperparameters ────────────────────────────────────────────────
        self.alpha          = 0.2    # learning rate
        self.gamma          = 0.95   # discount (more foresight than 0.9)
        self.epsilon        = 1.0
        self.epsilon_min    = 0.05
        self.epsilon_decay  = 0.997  # slower decay → longer exploration

        # ── target ─────────────────────────────────────────────────────────
        self.current_target  = np.array([1.0, 1.0])
        self.goal_threshold  = 1.5   # widened — easier to succeed early on

        # ── camera ─────────────────────────────────────────────────────────
        self.bridge         = CvBridge()
        self.target_visible = False
        self.visual_error   = 0.0    # negative = left, positive = right
        rospy.Subscriber('/Group1Bot/camera/image_raw', Image, self.image_callback)

        # ── Q-table ────────────────────────────────────────────────────────
        self.actions   = list(range(NUM_ACTIONS))
        self.q_table   = {}
        self.file_path = os.path.expanduser('~/q_table_group1.pkl')
        self.load_q_table()

        # ── ROS services / topics ──────────────────────────────────────────
        self.reset_proxy     = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self.set_state_proxy = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.get_model_proxy = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)

        self.vel_pub = rospy.Publisher('/Group1Bot/cmd_vel', Twist, queue_size=10)
        rospy.Subscriber('/Group1Bot/laser/scan', LaserScan, self.laser_callback)

        self.respawn_target()

        rospy.loginfo(
            "\n=== HOSPITAL PATROL BOT — Q-LEARNING ===\n"
            f"  State space : laser(243)×heading(8)×dist(8)×cam(4) = 62208\n"
            f"  Actions     : {NUM_ACTIONS}   α={self.alpha}  γ={self.gamma}\n"
            f"  Epsilon     : {self.epsilon:.2f} → {self.epsilon_min} "
            f"(decay {self.epsilon_decay})\n"
            f"  Goal radius : {self.goal_threshold}m\n"
            "=========================================\n"
        )

    # ── camera ────────────────────────────────────────────────────────────
    def image_callback(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            # Target marker is a GREEN cylinder → HSV hue ≈ 60° (green)
            mask = cv2.inRange(
                hsv,
                np.array([40, 80, 80]),
                np.array([80, 255, 255])
            )
            M = cv2.moments(mask)

            if M['m00'] > 300:
                cx                  = int(M['m10'] / M['m00'])
                self.visual_error   = (cx - 320) / 320.0  # +ve = right
                self.target_visible = True
            else:
                self.target_visible = False
                self.visual_error   = 0.0
        except Exception:
            pass

    # ── utils ─────────────────────────────────────────────────────────────
    def safe_min(self, ranges):
        vals = [x for x in ranges if math.isfinite(x) and x > 0.01]
        return min(vals) if vals else float('inf')

    def get_ground_truth(self):
        try:
            res = self.get_model_proxy('Group1Bot', 'world')
            self.prev_pos   = self.current_pos.copy()
            self.current_pos = np.array([res.pose.position.x, res.pose.position.y])
            q = res.pose.orientation
            _, _, self.robot_yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        except Exception:
            pass

    def argmax_random_tie(self, q):
        m = max(q)
        return random.choice([i for i, v in enumerate(q) if v == m])

    def default_q(self):
        return [0.0] * NUM_ACTIONS

    # ── state ─────────────────────────────────────────────────────────────
    def compute_state(self, msg, dist):
        # Five laser sectors: far-right, right, front, left, far-left
        # Laser spans -90° (idx 0) to +90° (idx 719); front = idx 288..431
        sector_starts = [0, 144, 288, 432, 576]
        # 3-level laser instead of binary: 0=danger(<0.5m), 1=caution(<1.2m), 2=clear
        def laser_level(ranges):
            d = self.safe_min(ranges)
            return 0 if d < 0.5 else (1 if d < 1.2 else 2)

        ls = tuple(laser_level(msg.ranges[s:s + 144]) for s in sector_starts)

        # Angle to target relative to robot heading → 8 bins
        angle_to_target = math.atan2(
            self.current_target[1] - self.current_pos[1],
            self.current_target[0] - self.current_pos[0],
        )
        err = angle_to_target - self.robot_yaw
        while err >  math.pi: err -= 2 * math.pi
        while err < -math.pi: err += 2 * math.pi
        hd = int(((err + math.pi) / (2 * math.pi)) * 8) % 8

        # Distance bin: 0–7 m (capped at 7)
        dist_bin = min(int(dist), 7)

        # Camera bin: 0=not visible, 1=visible-left, 2=visible-center, 3=visible-right
        if self.target_visible:
            cam = 1 if self.visual_error < -0.25 else (3 if self.visual_error > 0.25 else 2)
        else:
            cam = 0

        return str(ls + (hd, dist_bin, cam))

    # ── target respawn ─────────────────────────────────────────────────────
    def respawn_target(self):
        self.get_ground_truth()
        self.current_grace = self.spawn_grace_steps

        # Curriculum: start targets close (1.5 m), expand as agent improves.
        # Random angle guarantees the target is in open space around the robot
        # (not inside a wall), unlike fixed waypoint coordinates.
        max_dist = min(1.5 + self.episode_count * 0.01, 5.0)
        for _ in range(30):
            angle = random.uniform(0, 2 * math.pi)
            dist  = random.uniform(1.5, max_dist)
            x = self.current_pos[0] + dist * math.cos(angle)
            y = self.current_pos[1] + dist * math.sin(angle)
            if dist > 1.2:   # always place away from robot
                break

        self.current_target = np.array([x, y])

        state = ModelState()
        state.model_name        = 'target_marker'
        state.pose.position.x   = x
        state.pose.position.y   = y
        state.pose.position.z   = 0.05
        try:
            self.set_state_proxy(state)
        except Exception:
            pass

        rospy.loginfo(f"[TARGET] Anomaly at ({x:.2f}, {y:.2f}) "
                      f"dist={dist:.1f}m (curriculum max={max_dist:.1f}m)")

    # ── episode reset ──────────────────────────────────────────────────────
    def reset_episode(self):
        self.vel_pub.publish(Twist())

        robot = ModelState()
        robot.model_name            = 'Group1Bot'
        robot.pose.position.x       = 0.0
        robot.pose.position.y       = 10.0
        robot.pose.orientation.w    = 1.0
        try:
            self.set_state_proxy(robot)
        except Exception:
            self.reset_proxy()

        rospy.sleep(0.5)
        self.respawn_target()

    # ── Q-table I/O ───────────────────────────────────────────────────────
    def load_q_table(self):
        if not os.path.exists(self.file_path):
            return
        try:
            data = pickle.load(open(self.file_path, 'rb'))
            raw  = data.get('q_table', {})
            # Migrate old tables with fewer actions
            self.q_table = {
                k: (v + [0.0] * (NUM_ACTIONS - len(v)))[:NUM_ACTIONS]
                for k, v in raw.items()
            }
            self.epsilon        = data.get('epsilon',   1.0)
            self.episode_count  = data.get('episodes',  0)
            self.success_count  = data.get('successes', 0)
            rospy.loginfo(
                f"[Q-TABLE] Loaded {len(self.q_table)} states | "
                f"EP {self.episode_count} | EPS {self.epsilon:.3f}"
            )
        except Exception as e:
            rospy.logwarn(f"[Q-TABLE] Load failed ({e}), starting fresh.")

    def save_q_table(self):
        pickle.dump({
            'q_table':   self.q_table,
            'epsilon':   self.epsilon,
            'episodes':  self.episode_count,
            'successes': self.success_count,
        }, open(self.file_path, 'wb'))

    # ── logging ───────────────────────────────────────────────────────────
    def log_step(self, action, reward, dist, progress):
        names = {0: "FWD", 1: "LEFT", 2: "RIGHT", 3: "FWD+L", 4: "FWD+R"}
        sr = self.success_count / max(self.episode_count, 1) * 100
        rospy.loginfo_throttle(1.0,
            f"[EP {self.episode_count:04d}|ST {self.step_count:04d}] "
            f"{names[action]:6s}| DIST {dist:5.2f}m Δ{progress:+.3f} "
            f"R:{reward:7.2f} EPS:{self.epsilon:.3f} "
            f"CAM:{'Y' if self.target_visible else 'N'} "
            f"SR:{sr:.1f}% Q-sz:{len(self.q_table)}"
        )

    # ── main control loop (runs at laser rate ~20 Hz) ─────────────────────
    def laser_callback(self, msg):
        if self.current_grace > 0:
            self.current_grace -= 1

        self.get_ground_truth()
        dist = np.linalg.norm(self.current_pos - self.current_target)

        if self.last_dist is None:
            self.last_dist = dist

        progress = self.last_dist - dist  # positive = moving closer

        state = self.compute_state(msg, dist)

        # ── collision: wider ±45° front arc ───────────────────────────────
        front_ranges = msg.ranges[216:504]
        collision    = self.safe_min(front_ranges) < 0.45

        # ── stuck: robot barely moved for 1.5 s ───────────────────────────
        # Rolling-window stuck check: net displacement over last 50 steps (2.5 s).
        # Turning in place moves position only ~0 mm/step, so a single-step
        # threshold incorrectly treats normal steering as being stuck.
        self.pos_history.append(self.current_pos.copy())
        if len(self.pos_history) == 50:
            net_disp = np.linalg.norm(self.pos_history[-1] - self.pos_history[0])
            stuck = net_disp < 0.3   # < 30 cm net progress over 2.5 s = truly stuck
        else:
            stuck = False

        # ── reward ────────────────────────────────────────────────────────
        reward      = -0.3         # per-step cost — keep episodes short
        is_terminal = False

        if collision and self.current_grace <= 0:
            reward, is_terminal = -200.0, True

        elif stuck and self.current_grace <= 0:
            reward, is_terminal = -50.0, True

        elif dist < self.goal_threshold:
            # Time bonus: faster arrival → higher reward
            time_bonus = max(0, (1500 - self.step_count)) * 0.3
            reward     = 500.0 + time_bonus
            is_terminal = True

        elif self.step_count > 1500:
            reward, is_terminal = -50.0, True

        else:
            reward += progress * 300.0     # progress shaping

            # Proactive wall penalty: punish BEFORE collision so agent learns
            # to steer away while still in the "caution" zone (0.5–1.2 m)
            front_clear = self.safe_min(msg.ranges[216:504])
            if front_clear < 1.2:
                reward -= (1.2 - front_clear) * 5.0   # up to -6 at 0m, -0 at 1.2m

            if self.target_visible:
                reward += 2.0              # reward for seeing anomaly
                if abs(self.visual_error) < 0.25:
                    reward += 1.0          # bonus for centring camera on it

            reward -= 0.02 * dist          # gentle distance penalty

        self.total_ep_reward += reward
        self.step_count      += 1

        # ── Q update (Bellman) ────────────────────────────────────────────
        if self.last_state is not None:
            self.q_table.setdefault(self.last_state, self.default_q())
            self.q_table.setdefault(state,           self.default_q())
            max_next = max(self.q_table[state])

            self.q_table[self.last_state][self.last_action] += self.alpha * (
                reward
                + self.gamma * max_next
                - self.q_table[self.last_state][self.last_action]
            )

        # ── terminal handling ─────────────────────────────────────────────
        if is_terminal:
            if dist < self.goal_threshold:
                reason = "SUCCESS"
                self.success_count += 1
            elif collision:
                reason = "COLLISION"
                self.collision_count += 1
            elif stuck:
                reason = "STUCK"
            else:
                reason = "TIMEOUT"

            sr = self.success_count / max(self.episode_count + 1, 1) * 100
            rospy.logwarn(
                f"\n{'='*44}\n"
                f" EP {self.episode_count:04d} END → {reason}\n"
                f" Steps  : {self.step_count:4d} | Reward: {self.total_ep_reward:8.1f}\n"
                f" Dist   : {dist:.2f}m | Epsilon: {self.epsilon:.4f}\n"
                f" SR     : {sr:.1f}% ({self.success_count}/{self.episode_count+1})\n"
                f" Q-states: {len(self.q_table)}\n"
                f"{'='*44}"
            )

            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            self.episode_count += 1
            self.save_q_table()
            self.reset_episode()

            self.last_state      = None
            self.last_dist       = None
            self.pos_history.clear()
            self.total_ep_reward = 0.0
            self.step_count      = 0
            return

        # ── action selection (ε-greedy) ───────────────────────────────────
        self.q_table.setdefault(state, self.default_q())
        if random.random() < self.epsilon:
            action = random.choice(self.actions)
        else:
            action = self.argmax_random_tie(self.q_table[state])

        self.log_step(action, reward, dist, progress)

        # ── velocity command ──────────────────────────────────────────────
        move = Twist()
        if action == 0:   # forward
            move.linear.x  = 0.35
        elif action == 1: # spin left
            move.angular.z = 0.7
        elif action == 2: # spin right
            move.angular.z = -0.7
        elif action == 3: # forward + curve left
            move.linear.x  = 0.2
            move.angular.z = 0.45
        else:             # forward + curve right
            move.linear.x  = 0.2
            move.angular.z = -0.45

        self.vel_pub.publish(move)

        self.last_state  = state
        self.last_action = action
        self.last_dist   = dist


if __name__ == '__main__':
    QLearningAgent()
    rospy.spin()
