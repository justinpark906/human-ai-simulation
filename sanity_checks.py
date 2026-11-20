"""Sanity checks, physics unit tests, and matchup diagnostics for SumoEnv.

Run with: python sanity_checks.py
"""

import numpy as np

import sumo_env
from sumo_env import SumoEnv


SIDESTEP_CLOSE_DISTANCE = 80.0
NUMBER_OF_EPISODES_PER_MATCHUP = 500
BASE_RANDOM_SEED = 0

SHORT_EPISODE_LENGTH_THRESHOLD = 30
SHORT_EPISODE_FRACTION_WARNING_THRESHOLD = 0.5
TIMEOUT_FRACTION_WARNING_THRESHOLD = 0.5
SYMMETRIC_MATCHUP_WIN_RATE_TOLERANCE = 0.1


# ---------------------------------------------------------------------------
# Hand written agent policies for the matchup diagnostics.
# Each takes a flat observation (as produced by SumoEnv) and a numpy random
# generator, and returns a Discrete(5) action, exactly like a trained policy
# would be called.
# ---------------------------------------------------------------------------

def random_agent_policy(observation, random_generator):
    return int(random_generator.integers(0, 5))


def stationary_agent_policy(observation, random_generator):
    return sumo_env.ACTION_NO_PUSH


def charger_as_agent_policy(observation, random_generator):
    """The same chasing behavior as the scripted charger, but used to
    control the learning agent (so it charges the opponent instead)."""
    import scripted_opponents

    if random_generator.random() < scripted_opponents.CHARGER_RANDOM_ACTION_PROBABILITY:
        return int(random_generator.integers(0, 5))

    direction_to_opponent = observation[4:6]
    return scripted_opponents._direction_to_action(direction_to_opponent)


def sidestep_agent_policy(observation, random_generator):
    """When the opponent is close and approaching, sidesteps perpendicular
    to the opponent, choosing the perpendicular direction that moves the
    agent toward the ring center. Otherwise does nothing."""
    import scripted_opponents

    relative_position_vector = observation[4:6] * (2.0 * sumo_env.RING_RADIUS)
    relative_velocity_vector = observation[6:8] * (2.0 * sumo_env.MAX_SPEED)
    agent_position_vector = observation[0:2] * sumo_env.RING_RADIUS

    distance_to_opponent = np.linalg.norm(relative_position_vector)
    is_close = distance_to_opponent < SIDESTEP_CLOSE_DISTANCE
    is_approaching = np.dot(relative_position_vector, relative_velocity_vector) < 0.0

    if not (is_close and is_approaching):
        return sumo_env.ACTION_NO_PUSH

    unit_direction_to_opponent = relative_position_vector / max(distance_to_opponent, 1e-8)
    perpendicular_direction_one = np.array([-unit_direction_to_opponent[1], unit_direction_to_opponent[0]])
    perpendicular_direction_two = -perpendicular_direction_one

    direction_toward_center = -agent_position_vector
    if np.dot(perpendicular_direction_one, direction_toward_center) >= np.dot(perpendicular_direction_two, direction_toward_center):
        chosen_direction = perpendicular_direction_one
    else:
        chosen_direction = perpendicular_direction_two

    return scripted_opponents._direction_to_action(chosen_direction)


# ---------------------------------------------------------------------------
# Section 1: stable_baselines3 environment checker
# ---------------------------------------------------------------------------

def run_check_env():
    print("=" * 70)
    print("SECTION 1: stable_baselines3.common.env_checker.check_env")
    print("=" * 70)
    from stable_baselines3.common.env_checker import check_env

    environment = SumoEnv()
    check_env(environment)
    environment.close()
    print("check_env passed with no errors raised.\n")


# ---------------------------------------------------------------------------
# Section 2: physics unit tests
# ---------------------------------------------------------------------------

def test_speed_buildup_and_decay():
    environment = SumoEnv()
    environment.reset(seed=0)

    velocity = np.zeros(2, dtype=np.float64)
    observed_speeds = []
    for _ in range(200):
        velocity = environment._apply_acceleration_and_friction(velocity, sumo_env.ACTION_RIGHT)
        observed_speeds.append(np.linalg.norm(velocity))

    assert observed_speeds[0] < observed_speeds[10] < observed_speeds[-1] + 1e-6, \
        "speed should increase while pushing"
    assert abs(observed_speeds[-1] - sumo_env.MAX_SPEED) < 1e-3, \
        f"speed should plateau near MAX_SPEED, got {observed_speeds[-1]}"

    for _ in range(200):
        velocity = environment._apply_acceleration_and_friction(velocity, sumo_env.ACTION_NO_PUSH)
    assert np.linalg.norm(velocity) < 1e-3, \
        f"speed should decay to near zero after releasing the push, got {np.linalg.norm(velocity)}"

    environment.close()
    print("PASS: lone agent speeds up to MAX_SPEED under constant push and decays to near zero when released.")


def test_momentum_transfer_on_collision():
    environment = SumoEnv()
    environment.reset(seed=0)

    environment.learning_agent_position = np.array([-sumo_env.AGENT_RADIUS * 0.9, 0.0])
    environment.opponent_position = np.array([sumo_env.AGENT_RADIUS * 0.9, 0.0])
    environment.learning_agent_velocity = np.array([sumo_env.MAX_SPEED, 0.0])
    environment.opponent_velocity = np.array([0.0, 0.0])

    environment._resolve_collision()

    assert np.linalg.norm(environment.opponent_velocity) > 1.0, \
        "a stationary agent should be knocked back by a fast moving agent"
    assert np.linalg.norm(environment.learning_agent_velocity) < sumo_env.MAX_SPEED - 1.0, \
        "the moving agent should lose most of its velocity along the collision normal"

    environment.close()
    print("PASS: a moving agent transfers momentum to a stationary agent on collision.")


def test_circles_never_remain_overlapped_and_speeds_never_exceed_max():
    environment = SumoEnv(opponent_name="charger")
    random_generator = np.random.default_rng(123)

    for episode_index in range(10):
        environment.reset(seed=episode_index)
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action = int(random_generator.integers(0, 5))
            observation, reward, terminated, truncated, info = environment.step(action)

            distance_between_centers = np.linalg.norm(
                environment.learning_agent_position - environment.opponent_position
            )
            assert distance_between_centers >= 2.0 * sumo_env.AGENT_RADIUS - 1e-6, \
                f"circles are overlapped by {2.0 * sumo_env.AGENT_RADIUS - distance_between_centers}"

            learning_agent_speed = np.linalg.norm(environment.learning_agent_velocity)
            opponent_speed = np.linalg.norm(environment.opponent_velocity)
            assert learning_agent_speed <= sumo_env.MAX_SPEED + 1e-6, \
                f"learning agent speed {learning_agent_speed} exceeds MAX_SPEED"
            assert opponent_speed <= sumo_env.MAX_SPEED + 1e-6, \
                f"opponent speed {opponent_speed} exceeds MAX_SPEED"

    environment.close()
    print("PASS: circles never remain overlapped after a step, and speeds never exceed MAX_SPEED, "
          "over 10 random episodes.")


def run_physics_unit_tests():
    print("=" * 70)
    print("SECTION 2: physics unit tests")
    print("=" * 70)
    test_speed_buildup_and_decay()
    test_momentum_transfer_on_collision()
    test_circles_never_remain_overlapped_and_speeds_never_exceed_max()
    print()


# ---------------------------------------------------------------------------
# Section 3 and 4: matchup statistics and recorded videos
# ---------------------------------------------------------------------------

def run_matchup(opponent_name, agent_policy_function, number_of_episodes, base_seed):
    environment = SumoEnv(condition="baseline", opponent_name=opponent_name, render_mode=None)

    outcomes = []
    episode_lengths = []
    for episode_index in range(number_of_episodes):
        observation, info = environment.reset(seed=base_seed + episode_index)
        agent_random_generator = np.random.default_rng(base_seed + episode_index + 1_000_000)

        terminated = False
        truncated = False
        while not (terminated or truncated):
            action = agent_policy_function(observation, agent_random_generator)
            observation, reward, terminated, truncated, info = environment.step(action)

        outcomes.append(info["outcome"])
        episode_lengths.append(info["episode_length"])

    environment.close()
    return outcomes, episode_lengths


def summarize_matchup(matchup_name, outcomes, episode_lengths):
    number_of_episodes = len(outcomes)
    win_rate = outcomes.count("win") / number_of_episodes
    loss_rate = outcomes.count("loss") / number_of_episodes
    draw_rate = outcomes.count("draw") / number_of_episodes

    episode_lengths_array = np.array(episode_lengths)
    print(f"--- {matchup_name} ---")
    print(f"  win rate:  {win_rate:.3f}")
    print(f"  loss rate: {loss_rate:.3f}")
    print(f"  draw rate: {draw_rate:.3f}")
    print(f"  episode length: mean={episode_lengths_array.mean():.1f}  "
          f"std={episode_lengths_array.std():.1f}  "
          f"min={episode_lengths_array.min()}  "
          f"p25={np.percentile(episode_lengths_array, 25):.0f}  "
          f"median={np.percentile(episode_lengths_array, 50):.0f}  "
          f"p75={np.percentile(episode_lengths_array, 75):.0f}  "
          f"max={episode_lengths_array.max()}")
    print()

    return {
        "win_rate": win_rate,
        "loss_rate": loss_rate,
        "draw_rate": draw_rate,
        "mean_episode_length": float(episode_lengths_array.mean()),
        "short_episode_fraction": float(np.mean(episode_lengths_array < SHORT_EPISODE_LENGTH_THRESHOLD)),
        "timeout_fraction": float(np.mean(episode_lengths_array >= sumo_env.MAX_STEPS)),
    }


def run_matchup_statistics_and_videos(save_videos):
    print("=" * 70)
    print("SECTION 3 and 4: matchup statistics (and recorded mp4s) over "
          f"{NUMBER_OF_EPISODES_PER_MATCHUP} episodes each")
    print("=" * 70)

    matchups = [
        ("random agent vs charger", random_agent_policy, "charger"),
        ("agent that never moves vs charger", stationary_agent_policy, "charger"),
        ("charger logic used as the agent vs charger", charger_as_agent_policy, "charger"),
        ("sidestep agent vs charger", sidestep_agent_policy, "charger"),
    ]

    summaries = {}
    for matchup_name, agent_policy_function, opponent_name in matchups:
        outcomes, episode_lengths = run_matchup(
            opponent_name, agent_policy_function, NUMBER_OF_EPISODES_PER_MATCHUP, BASE_RANDOM_SEED
        )
        summaries[matchup_name] = summarize_matchup(matchup_name, outcomes, episode_lengths)

        if save_videos:
            video_file_name = "sanity_check_video_" + matchup_name.replace(" ", "_") + ".mp4"
            video_environment = SumoEnv(condition="baseline", opponent_name=opponent_name, render_mode="rgb_array")
            video_random_generator = np.random.default_rng(BASE_RANDOM_SEED)

            def video_policy_function(observation, _random_generator=video_random_generator,
                                       _policy_function=agent_policy_function):
                return _policy_function(observation, _random_generator)

            sumo_env.run_episode_and_save_video(video_environment, video_policy_function, video_file_name)
            video_environment.close()
            print(f"  saved video: {video_file_name}")

    print()
    return summaries


# ---------------------------------------------------------------------------
# Section 5: tuning diagnostics
# ---------------------------------------------------------------------------

def print_tuning_diagnostics(summaries):
    print("=" * 70)
    print("SECTION 5: tuning diagnostics")
    print("=" * 70)

    flags_raised = False

    charger_vs_charger_summary = summaries["charger logic used as the agent vs charger"]
    if charger_vs_charger_summary["short_episode_fraction"] > SHORT_EPISODE_FRACTION_WARNING_THRESHOLD:
        flags_raised = True
        print("FLAG: most episodes in the symmetric matchup end in under "
              f"{SHORT_EPISODE_LENGTH_THRESHOLD} steps. Pushing may be too strong relative to the "
              "ring size and agent radius. Consider lowering ACCELERATION_MAGNITUDE or MAX_SPEED, "
              "or increasing RING_RADIUS.")

    if charger_vs_charger_summary["timeout_fraction"] > TIMEOUT_FRACTION_WARNING_THRESHOLD:
        flags_raised = True
        print("FLAG: most episodes in the symmetric matchup time out. Pushing may be too weak, or "
              "the ring may be too large relative to agent speed. Consider raising "
              "ACCELERATION_MAGNITUDE or MAX_SPEED, or lowering RING_RADIUS or MAX_STEPS.")

    win_rate_deviation_from_half = abs(charger_vs_charger_summary["win_rate"] - 0.5)
    draw_rate_is_dominant = charger_vs_charger_summary["draw_rate"] > 0.5
    if win_rate_deviation_from_half > SYMMETRIC_MATCHUP_WIN_RATE_TOLERANCE and not draw_rate_is_dominant:
        flags_raised = True
        print(f"FLAG: the symmetric matchup (charger logic vs charger) has a win rate of "
              f"{charger_vs_charger_summary['win_rate']:.3f}, far from 0.5. This suggests a bias in "
              "starting positions or in the physics/collision resolution (e.g. the learning agent's "
              "updates are not applied symmetrically with the opponent's).")

    if not flags_raised:
        print("No tuning problems flagged: episode lengths are reasonably distributed and the "
              "symmetric matchup is close to 50/50.")
    print()


def main():
    run_check_env()
    run_physics_unit_tests()
    summaries = run_matchup_statistics_and_videos(save_videos=True)
    print_tuning_diagnostics(summaries)


if __name__ == "__main__":
    main()
