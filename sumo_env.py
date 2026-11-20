"""A minimal 2D sumo environment for reinforcement learning experiments.

Two circular agents push each other around inside a circular ring. The
learning agent is controlled by the policy; the opponent is controlled by
one of the scripted policies in scripted_opponents.py. The environment
supports three reward conditions ("baseline", "hard_rule", "soft_penalty")
that differ only in how they treat a STALLING rule: the learning agent must
make contact with the opponent at least once every STALL_LIMIT steps.
"""

import math

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces

import scripted_opponents


# ---------------------------------------------------------------------------
# Physics and game constants (all tunable values live here)
# ---------------------------------------------------------------------------

RING_RADIUS = 300.0
AGENT_RADIUS = 20.0
AGENT_MASS = 1.0

ACCELERATION_MAGNITUDE = 1.2
FRICTION_COEFFICIENT = 0.08
MAX_SPEED = 6.0

STARTING_DISTANCE_FROM_CENTER = 150.0

MAX_STEPS = 600
STALL_LIMIT = 100
DEFAULT_SOFT_PENALTY_WEIGHT = 0.01

EDGE_ZONE_FRACTION = 0.2

# Rendering constants
RENDER_WINDOW_SIZE = 700
RENDER_MARGIN = 40
RENDER_BACKGROUND_COLOR = (245, 245, 240)
RENDER_RING_COLOR = (40, 40, 40)
RENDER_LEARNING_AGENT_COLOR = (200, 30, 30)
RENDER_OPPONENT_COLOR = (30, 60, 200)
RENDER_TRAIL_LENGTH = 25
RENDER_TEXT_COLOR = (20, 20, 20)

# Action space: no push, up, down, left, right
ACTION_NO_PUSH = 0
ACTION_UP = 1
ACTION_DOWN = 2
ACTION_LEFT = 3
ACTION_RIGHT = 4

ACTION_TO_DIRECTION = {
    ACTION_NO_PUSH: np.array([0.0, 0.0]),
    ACTION_UP: np.array([0.0, 1.0]),
    ACTION_DOWN: np.array([0.0, -1.0]),
    ACTION_LEFT: np.array([-1.0, 0.0]),
    ACTION_RIGHT: np.array([1.0, 0.0]),
}

# Observation normalization scales
POSITION_NORMALIZATION_SCALE = RING_RADIUS
VELOCITY_NORMALIZATION_SCALE = MAX_SPEED
RELATIVE_POSITION_NORMALIZATION_SCALE = 2.0 * RING_RADIUS
EDGE_DISTANCE_NORMALIZATION_SCALE = RING_RADIUS

VALID_CONDITIONS = ("baseline", "hard_rule", "soft_penalty")
VALID_OPPONENT_NAMES = ("charger", "wanderer", "defender")


def check_stalling_rule_violation(steps_since_last_contact):
    """Returns True if the learning agent has gone too long without contact.

    This is the single place that defines the "stalling" rule so a
    different rule can be substituted later without touching step logic.
    """
    return steps_since_last_contact > STALL_LIMIT


class SumoEnv(gym.Env):
    """A two-agent circular sumo environment with one learning agent.

    Observation layout (flat float32 array of length 10), all values
    roughly normalized to [-1, 1] and all relative to the learning agent:
        index 0: learning agent position x (divided by RING_RADIUS)
        index 1: learning agent position y (divided by RING_RADIUS)
        index 2: learning agent velocity x (divided by MAX_SPEED)
        index 3: learning agent velocity y (divided by MAX_SPEED)
        index 4: opponent position x relative to learning agent (divided by 2*RING_RADIUS)
        index 5: opponent position y relative to learning agent (divided by 2*RING_RADIUS)
        index 6: opponent velocity x relative to learning agent (divided by 2*MAX_SPEED)
        index 7: opponent velocity y relative to learning agent (divided by 2*MAX_SPEED)
        index 8: learning agent distance to the ring edge (divided by RING_RADIUS)
        index 9: opponent distance to the ring edge (divided by RING_RADIUS)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, condition="baseline", opponent_name="charger",
                 soft_penalty_weight=DEFAULT_SOFT_PENALTY_WEIGHT, render_mode=None):
        super().__init__()

        if condition not in VALID_CONDITIONS:
            raise ValueError(f"condition must be one of {VALID_CONDITIONS}, got {condition!r}")
        if opponent_name not in VALID_OPPONENT_NAMES:
            raise ValueError(f"opponent_name must be one of {VALID_OPPONENT_NAMES}, got {opponent_name!r}")
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"render_mode must be one of {self.metadata['render_modes']} or None")

        self.condition = condition
        self.opponent_name = opponent_name
        self.soft_penalty_weight = soft_penalty_weight
        self.render_mode = render_mode

        self.opponent_policy_function = getattr(scripted_opponents, opponent_name)

        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=-2.0, high=2.0, shape=(10,), dtype=np.float32)

        self.learning_agent_position = np.zeros(2, dtype=np.float64)
        self.learning_agent_velocity = np.zeros(2, dtype=np.float64)
        self.opponent_position = np.zeros(2, dtype=np.float64)
        self.opponent_velocity = np.zeros(2, dtype=np.float64)

        self.current_step_count = 0
        self.steps_since_last_contact = 0
        self.total_contact_count = 0
        self.steps_near_edge_count = 0
        self.rule_violation_step_count = 0

        self.learning_agent_position_history = []
        self.opponent_position_history = []

        self.opponent_random_generator = np.random.default_rng()
        self.opponent_internal_state = {}

        self.pygame_window_surface = None
        self.pygame_clock = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        starting_angle = self.np_random.uniform(0.0, 2.0 * math.pi)
        self.learning_agent_position = np.array(
            [STARTING_DISTANCE_FROM_CENTER * math.cos(starting_angle),
             STARTING_DISTANCE_FROM_CENTER * math.sin(starting_angle)],
            dtype=np.float64,
        )
        self.opponent_position = -self.learning_agent_position.copy()

        self.learning_agent_velocity = np.zeros(2, dtype=np.float64)
        self.opponent_velocity = np.zeros(2, dtype=np.float64)

        self.current_step_count = 0
        self.steps_since_last_contact = 0
        self.total_contact_count = 0
        self.steps_near_edge_count = 0
        self.rule_violation_step_count = 0

        self.learning_agent_position_history = [self.learning_agent_position.copy()]
        self.opponent_position_history = [self.opponent_position.copy()]

        opponent_seed = int(self.np_random.integers(0, 2**31 - 1))
        self.opponent_random_generator = np.random.default_rng(opponent_seed)
        self.opponent_internal_state = {}

        observation = self._build_observation()
        info = {
            "step_count": self.current_step_count,
            "contact_this_step": False,
            "steps_since_last_contact": self.steps_since_last_contact,
            "rule_violated_this_step": False,
        }

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        opponent_action = self.opponent_policy_function(self, self.opponent_random_generator)

        self.learning_agent_velocity = self._apply_acceleration_and_friction(
            self.learning_agent_velocity, action
        )
        self.opponent_velocity = self._apply_acceleration_and_friction(
            self.opponent_velocity, opponent_action
        )

        self.learning_agent_position = self.learning_agent_position + self.learning_agent_velocity
        self.opponent_position = self.opponent_position + self.opponent_velocity

        contact_this_step = self._resolve_collision()
        if contact_this_step:
            self.total_contact_count += 1
            self.steps_since_last_contact = 0
        else:
            self.steps_since_last_contact += 1

        self.current_step_count += 1

        learning_agent_distance_to_edge = RING_RADIUS - np.linalg.norm(self.learning_agent_position)
        opponent_distance_to_edge = RING_RADIUS - np.linalg.norm(self.opponent_position)

        if learning_agent_distance_to_edge < EDGE_ZONE_FRACTION * RING_RADIUS:
            self.steps_near_edge_count += 1

        rule_violated_this_step = check_stalling_rule_violation(self.steps_since_last_contact)
        if rule_violated_this_step:
            self.rule_violation_step_count += 1

        self.learning_agent_position_history.append(self.learning_agent_position.copy())
        self.opponent_position_history.append(self.opponent_position.copy())

        learning_agent_is_out = learning_agent_distance_to_edge < 0.0
        opponent_is_out = opponent_distance_to_edge < 0.0
        timed_out = self.current_step_count >= MAX_STEPS

        terminated = False
        truncated = False
        outcome = None
        reward = 0.0

        if self.condition == "hard_rule" and rule_violated_this_step:
            terminated = True
            outcome = "rule_loss"
            reward = -1.0
        elif learning_agent_is_out and opponent_is_out:
            terminated = True
            outcome = "draw"
            reward = 0.0
        elif learning_agent_is_out:
            terminated = True
            outcome = "loss"
            reward = -1.0
        elif opponent_is_out:
            terminated = True
            outcome = "win"
            reward = 1.0
        elif timed_out:
            truncated = True
            outcome = "draw"
            reward = 0.0

        if self.condition == "soft_penalty" and rule_violated_this_step:
            reward -= self.soft_penalty_weight

        info = {
            "step_count": self.current_step_count,
            "contact_this_step": contact_this_step,
            "steps_since_last_contact": self.steps_since_last_contact,
            "rule_violated_this_step": rule_violated_this_step,
        }

        if terminated or truncated:
            info["outcome"] = outcome
            info["episode_length"] = self.current_step_count
            info["total_contacts"] = self.total_contact_count
            info["steps_near_edge"] = self.steps_near_edge_count
            info["rule_violation_steps"] = self.rule_violation_step_count
            info["learning_agent_position_history"] = list(self.learning_agent_position_history)
            info["opponent_position_history"] = list(self.opponent_position_history)

        observation = self._build_observation()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info

    def _apply_acceleration_and_friction(self, current_velocity, action):
        push_direction = ACTION_TO_DIRECTION[action]
        velocity_after_push = current_velocity + push_direction * ACCELERATION_MAGNITUDE
        velocity_after_friction = velocity_after_push * (1.0 - FRICTION_COEFFICIENT)
        return self._clip_speed_to_max(velocity_after_friction)

    def _resolve_collision(self):
        separation_vector = self.opponent_position - self.learning_agent_position
        distance_between_centers = np.linalg.norm(separation_vector)
        minimum_allowed_distance = 2.0 * AGENT_RADIUS

        if distance_between_centers >= minimum_allowed_distance:
            return False

        if distance_between_centers < 1e-8:
            collision_normal = np.array([1.0, 0.0])
            distance_between_centers = 1e-8
        else:
            collision_normal = separation_vector / distance_between_centers

        overlap_amount = minimum_allowed_distance - distance_between_centers
        self.learning_agent_position = self.learning_agent_position - collision_normal * (overlap_amount / 2.0)
        self.opponent_position = self.opponent_position + collision_normal * (overlap_amount / 2.0)

        learning_agent_normal_speed = np.dot(self.learning_agent_velocity, collision_normal)
        opponent_normal_speed = np.dot(self.opponent_velocity, collision_normal)

        learning_agent_tangential_velocity = self.learning_agent_velocity - learning_agent_normal_speed * collision_normal
        opponent_tangential_velocity = self.opponent_velocity - opponent_normal_speed * collision_normal

        self.learning_agent_velocity = learning_agent_tangential_velocity + opponent_normal_speed * collision_normal
        self.opponent_velocity = opponent_tangential_velocity + learning_agent_normal_speed * collision_normal

        self.learning_agent_velocity = self._clip_speed_to_max(self.learning_agent_velocity)
        self.opponent_velocity = self._clip_speed_to_max(self.opponent_velocity)

        return True

    def _clip_speed_to_max(self, velocity):
        speed = np.linalg.norm(velocity)
        if speed > MAX_SPEED:
            return velocity * (MAX_SPEED / speed)
        return velocity

    def _build_observation(self):
        relative_opponent_position = self.opponent_position - self.learning_agent_position
        relative_opponent_velocity = self.opponent_velocity - self.learning_agent_velocity

        learning_agent_distance_to_edge = RING_RADIUS - np.linalg.norm(self.learning_agent_position)
        opponent_distance_to_edge = RING_RADIUS - np.linalg.norm(self.opponent_position)

        observation = np.array(
            [
                self.learning_agent_position[0] / POSITION_NORMALIZATION_SCALE,
                self.learning_agent_position[1] / POSITION_NORMALIZATION_SCALE,
                self.learning_agent_velocity[0] / VELOCITY_NORMALIZATION_SCALE,
                self.learning_agent_velocity[1] / VELOCITY_NORMALIZATION_SCALE,
                relative_opponent_position[0] / RELATIVE_POSITION_NORMALIZATION_SCALE,
                relative_opponent_position[1] / RELATIVE_POSITION_NORMALIZATION_SCALE,
                relative_opponent_velocity[0] / (2.0 * VELOCITY_NORMALIZATION_SCALE),
                relative_opponent_velocity[1] / (2.0 * VELOCITY_NORMALIZATION_SCALE),
                learning_agent_distance_to_edge / EDGE_DISTANCE_NORMALIZATION_SCALE,
                opponent_distance_to_edge / EDGE_DISTANCE_NORMALIZATION_SCALE,
            ],
            dtype=np.float32,
        )
        return observation

    def _world_to_screen(self, world_position):
        pixels_per_unit = (RENDER_WINDOW_SIZE - 2.0 * RENDER_MARGIN) / (2.0 * RING_RADIUS)
        screen_x = RENDER_WINDOW_SIZE / 2.0 + world_position[0] * pixels_per_unit
        screen_y = RENDER_WINDOW_SIZE / 2.0 - world_position[1] * pixels_per_unit
        return int(screen_x), int(screen_y)

    def _render_frame(self):
        if self.pygame_window_surface is None:
            pygame.init()
            pygame.font.init()
            if self.render_mode == "human":
                self.pygame_window_surface = pygame.display.set_mode(
                    (RENDER_WINDOW_SIZE, RENDER_WINDOW_SIZE)
                )
            else:
                self.pygame_window_surface = pygame.Surface(
                    (RENDER_WINDOW_SIZE, RENDER_WINDOW_SIZE)
                )
            self.pygame_clock = pygame.time.Clock()

        canvas = pygame.Surface((RENDER_WINDOW_SIZE, RENDER_WINDOW_SIZE))
        canvas.fill(RENDER_BACKGROUND_COLOR)

        ring_center_screen = self._world_to_screen(np.array([0.0, 0.0]))
        pixels_per_unit = (RENDER_WINDOW_SIZE - 2.0 * RENDER_MARGIN) / (2.0 * RING_RADIUS)
        ring_radius_pixels = int(RING_RADIUS * pixels_per_unit)
        pygame.draw.circle(canvas, RENDER_RING_COLOR, ring_center_screen, ring_radius_pixels, width=3)

        trail_history_pairs = [
            (self.learning_agent_position_history, RENDER_LEARNING_AGENT_COLOR),
            (self.opponent_position_history, RENDER_OPPONENT_COLOR),
        ]
        for position_history, trail_color in trail_history_pairs:
            recent_positions = position_history[-RENDER_TRAIL_LENGTH:]
            for trail_index, trail_position in enumerate(recent_positions):
                trail_alpha_fraction = (trail_index + 1) / max(len(recent_positions), 1)
                faded_color = tuple(int(channel * trail_alpha_fraction + 245 * (1 - trail_alpha_fraction))
                                    for channel in trail_color)
                pygame.draw.circle(canvas, faded_color, self._world_to_screen(trail_position), 3)

        agent_radius_pixels = int(AGENT_RADIUS * pixels_per_unit)
        pygame.draw.circle(canvas, RENDER_LEARNING_AGENT_COLOR,
                            self._world_to_screen(self.learning_agent_position), agent_radius_pixels)
        pygame.draw.circle(canvas, RENDER_OPPONENT_COLOR,
                            self._world_to_screen(self.opponent_position), agent_radius_pixels)

        status_font = pygame.font.SysFont("arial", 18)
        status_text = f"step {self.current_step_count}   condition: {self.condition}"
        text_surface = status_font.render(status_text, True, RENDER_TEXT_COLOR)
        canvas.blit(text_surface, (10, 10))

        if self.render_mode == "human":
            self.pygame_window_surface.blit(canvas, (0, 0))
            pygame.event.pump()
            pygame.display.flip()
            self.pygame_clock.tick(self.metadata["render_fps"])
        else:
            self.pygame_window_surface = canvas

        return canvas

    def render(self):
        if self.render_mode is None:
            return None
        canvas = self._render_frame()
        if self.render_mode == "rgb_array":
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))
        return None

    def close(self):
        if self.pygame_window_surface is not None:
            pygame.display.quit()
            pygame.quit()
            self.pygame_window_surface = None


def run_episode_and_save_video(environment, policy_function, output_file_path, maximum_steps=MAX_STEPS):
    """Runs one episode with the given policy function and saves it as an mp4.

    policy_function takes the observation and returns a Discrete(5) action.
    Requires the environment to have been constructed with render_mode="rgb_array".
    """
    import imageio

    video_frames = []
    observation, info = environment.reset()
    video_frames.append(environment.render())

    for _ in range(maximum_steps):
        action = policy_function(observation)
        observation, reward, terminated, truncated, info = environment.step(action)
        video_frames.append(environment.render())
        if terminated or truncated:
            break

    imageio.mimsave(output_file_path, video_frames, fps=environment.metadata["render_fps"])
    return info
