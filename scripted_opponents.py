"""Scripted opponent policies for the sumo environment.

Each policy is a function that takes the environment (so it can read
positions, velocities, and the ring radius) and a numpy random generator,
and returns an action in the same Discrete(5) action space the learning
agent uses: 0 = no push, 1 = up, 2 = down, 3 = left, 4 = right.

The environment stores a per-episode dictionary at
environment.opponent_internal_state that a policy may use to remember
things (such as the wanderer's current direction) across steps; this
dictionary is reset to empty at the start of every episode.
"""

import numpy as np

import sumo_env


CHARGER_RANDOM_ACTION_PROBABILITY = 0.1

WANDERER_MINIMUM_HOLD_STEPS = 15
WANDERER_MAXIMUM_HOLD_STEPS = 60
WANDERER_EDGE_STEER_FRACTION = 0.25

DEFENDER_ENGAGEMENT_DISTANCE = 120.0
DEFENDER_CENTER_RETURN_DISTANCE = 30.0


def _direction_to_action(direction_vector):
    """Converts a continuous direction into the closest Discrete(5) action."""
    horizontal_component, vertical_component = direction_vector[0], direction_vector[1]

    if abs(horizontal_component) < 1e-6 and abs(vertical_component) < 1e-6:
        return sumo_env.ACTION_NO_PUSH

    if abs(horizontal_component) >= abs(vertical_component):
        return sumo_env.ACTION_RIGHT if horizontal_component > 0 else sumo_env.ACTION_LEFT
    else:
        return sumo_env.ACTION_UP if vertical_component > 0 else sumo_env.ACTION_DOWN


def charger(environment, random_generator):
    """Moves toward the learning agent, occasionally acting randomly."""
    if random_generator.random() < CHARGER_RANDOM_ACTION_PROBABILITY:
        return int(random_generator.integers(0, 5))

    direction_toward_learning_agent = environment.learning_agent_position - environment.opponent_position
    return _direction_to_action(direction_toward_learning_agent)


def wanderer(environment, random_generator):
    """Holds a random direction for a random number of steps, steering back
    toward the center when close to the ring edge."""
    internal_state = environment.opponent_internal_state

    distance_from_center = np.linalg.norm(environment.opponent_position)
    near_edge = distance_from_center > (1.0 - WANDERER_EDGE_STEER_FRACTION) * sumo_env.RING_RADIUS

    if near_edge:
        direction_toward_center = -environment.opponent_position
        internal_state["wanderer_remaining_hold_steps"] = 0
        return _direction_to_action(direction_toward_center)

    remaining_hold_steps = internal_state.get("wanderer_remaining_hold_steps", 0)
    if remaining_hold_steps <= 0:
        random_angle = random_generator.uniform(0.0, 2.0 * np.pi)
        internal_state["wanderer_current_direction"] = np.array(
            [np.cos(random_angle), np.sin(random_angle)]
        )
        internal_state["wanderer_remaining_hold_steps"] = int(
            random_generator.integers(WANDERER_MINIMUM_HOLD_STEPS, WANDERER_MAXIMUM_HOLD_STEPS + 1)
        )

    internal_state["wanderer_remaining_hold_steps"] -= 1
    return _direction_to_action(internal_state["wanderer_current_direction"])


def defender(environment, random_generator):
    """Stays near the center, only pushing toward the agent when it is close."""
    distance_to_learning_agent = np.linalg.norm(
        environment.learning_agent_position - environment.opponent_position
    )

    if distance_to_learning_agent <= DEFENDER_ENGAGEMENT_DISTANCE:
        direction_toward_learning_agent = environment.learning_agent_position - environment.opponent_position
        return _direction_to_action(direction_toward_learning_agent)

    distance_from_center = np.linalg.norm(environment.opponent_position)
    if distance_from_center > DEFENDER_CENTER_RETURN_DISTANCE:
        direction_toward_center = -environment.opponent_position
        return _direction_to_action(direction_toward_center)

    return sumo_env.ACTION_NO_PUSH
