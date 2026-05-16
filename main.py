import gymnasium as gym
import pygad
import time

env = gym.make("Taxi-v4")


def fitness_func(ga_instance, solution, solution_idx):
    # THE STANDARDIZED TEST: These 3 seeds ensure the exact same 3 starting scenarios are tested every generation.
    # This completely eliminates "lucky spawns" breaking the algorithm.
    test_seeds = [42, 105, 999]
    total_score = 0
    zone_coordinates = [(0, 0), (0, 4), (4, 0), (4, 3)]
    for seed in test_seeds:
        state, info = env.reset(seed=seed)
        episode_reward = 0
        done = False
        truncated = False

        step_count = 0
        wall_bumps = 0
        picked_up = False

        taxi_row, taxi_col, pass_idx, dest_idx = list(env.unwrapped.decode(state))
        target_row, target_col = zone_coordinates[pass_idx] if pass_idx < 4 else zone_coordinates[dest_idx]
        closest_dist = abs(taxi_row - target_row) + abs(taxi_col - target_col)
        # 100 steps to allow enough time to finish, but not enough to waste training time
        while not done and not truncated and step_count < 100:
            action = int(round(solution[state]))
            action = max(0, min(5, action))

            # --- PRECISE WALL BUMP DETECTION VIA ACTION MASK ---
            # The docs state info["action_mask"] specifies if an action is valid.
            # If the mask is 0 for our chosen movement action, it's a wall bump!
            if info["action_mask"][action] == 0 and action in [0, 1, 2, 3]:
                wall_bumps += 1
                episode_reward -= 5  # Active punishment for hitting a wall

            # Take the step
            state, reward, done, truncated, info = env.step(action)
            new_row, new_col, new_pass, new_dest = list(env.unwrapped.decode(state))

            #upon pickup, change target
            target_row, target_col = zone_coordinates[new_pass] if new_pass < 4 else zone_coordinates[new_dest]
            current_dist = abs(new_row - target_row) + abs(new_col - target_col)
            if current_dist < closest_dist:
                episode_reward += 5
                closest_dist = current_dist
            # Did we pick up the passenger? (Reward = -1 for step, but state changes)
            # We can decode the state to check if passenger is in taxi (index 4)
            pass_loc = list(env.unwrapped.decode(state))[2]
            if pass_loc == 4 and not picked_up:
                episode_reward += 50  # Milestone reward for picking up
                picked_up = True

            episode_reward += reward  # Standard Gym rewards (-1 step, -10 bad drop, +20 success)
            step_count += 1

        # --- TEACHER'S FINAL GRADING ---
        # If the episode ended successfully (reward for dropoff is 20)
        if reward == 20:
            if wall_bumps == 0:
                episode_reward += 50
            elif wall_bumps <= 3:
                episode_reward += 20

        total_score += episode_reward

    return total_score / len(test_seeds)


def on_generation(ga_instance):
    if ga_instance.generations_completed % 100 == 0:
        current_best_score = ga_instance.best_solution()[1]
        print(
            f"Generation {ga_instance.generations_completed} / {ga_instance.num_generations} | Best Score: {current_best_score}")


# Setup the Genetic Algorithm
ga_instance = pygad.GA(
    num_generations=500,  # Increased because evolving 500 exact states takes time
    num_parents_mating=50,
    fitness_func=fitness_func,
    sol_per_pop=100,
    num_genes=500,
    init_range_low=0,
    init_range_high=5,
    gene_type=int,
    mutation_probability=0.25,  # Balanced mutation
    on_generation=on_generation,
    parallel_processing=["thread", 14]
)

print("Starting training with Fixed Seeds...")
ga_instance.run()

solution, solution_fitness, solution_idx = ga_instance.best_solution()
print(f"Training Complete! Best score: {solution_fitness}")

# --- TEST RUN ---
# We use one of the training seeds to watch it execute the path it learned
test_env = gym.make("Taxi-v4", render_mode="human")
state, info = test_env.reset(seed=42)

done = False
truncated = False
total_test_reward = 0

while not done and not truncated:
    action = int(round(solution[state]))
    action = max(0, min(5, action))

    state, reward, done, truncated, info = test_env.step(action)
    total_test_reward += reward
    time.sleep(0.2)

print(f"Final Test Run Score: {total_test_reward}")
test_env.close()