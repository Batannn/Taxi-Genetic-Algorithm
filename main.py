import gymnasium as gym
import pygad
import time
import numpy as np  # Needed for the override logic

env = gym.make("Taxi-v4")

#dictionary to track exact opposite directions (0:South, 1:North, 2:East, 3:West)
OPPOSITES = {0: 1, 1: 0, 2: 3, 3: 2, 4: -1, 5: -1}


def fitness_func(ga_instance, solution, solution_idx):
    test_seeds = [42, 105, 999]
    total_score = 0
    zone_coordinates = [(0, 0), (0, 4), (4, 0), (4, 3)]

    for seed in test_seeds:
        state, info = env.reset(seed=seed)
        episode_reward = 0
        done = False
        truncated = False

        step_count = 0
        picked_up = False
        last_action = -1 #movement tracking

        taxi_row, taxi_col, pass_idx, dest_idx = list(env.unwrapped.decode(state))
        target_row, target_col = zone_coordinates[pass_idx] if pass_idx < 4 else zone_coordinates[dest_idx]
        closest_dist = abs(taxi_row - target_row) + abs(taxi_col - target_col)

        while not done and not truncated and step_count < 100:
            #ask ga what to do
            intended_action = int(round(solution[state]))
            intended_action = max(0, min(5, intended_action))
            final_action = intended_action

            override_occurred = False

            #ban wall bump
            if intended_action < 4 and info["action_mask"][intended_action] == 0:
                episode_reward -= 10  # Punish the bad gene
                override_occurred = True

            #ban loop
            elif intended_action < 4 and intended_action == OPPOSITES.get(last_action, -1):
                episode_reward -= 10  # Punish the looping gene
                override_occurred = True

           #instructor
            if override_occurred:
                # Find all physically valid moves that won't hit a wall
                valid_moves = [a for a in range(4) if info["action_mask"][a] == 1]

                # Remove the backward step from valid moves
                if OPPOSITES.get(last_action, -1) in valid_moves and len(valid_moves) > 1:
                    valid_moves.remove(OPPOSITES[last_action])

                # Force the taxi to take a valid action
                final_action = np.random.choice(valid_moves)

            #step
            state, reward, done, truncated, info = env.step(final_action)
            new_row, new_col, new_pass, new_dest = list(env.unwrapped.decode(state))

            #update Breadcrumbs
            target_row, target_col = zone_coordinates[new_pass] if new_pass < 4 else zone_coordinates[new_dest]
            current_dist = abs(new_row - target_row) + abs(new_col - target_col)

            if current_dist < closest_dist:
                episode_reward += 5
                closest_dist = current_dist

            #milestone Rewards
            pass_loc = list(env.unwrapped.decode(state))[2]
            if pass_loc == 4 and not picked_up:
                episode_reward += 50
                picked_up = True
                closest_dist = 999

            episode_reward += reward
            step_count += 1
            last_action = final_action#remember move

        #final grading
        if reward == 20:
            episode_reward += 100

        total_score += episode_reward

    return total_score / len(test_seeds)


def on_generation(ga_instance):
    if ga_instance.generations_completed % 100 == 0:
        current_best_score = ga_instance.best_solution()[1]
        print(
            f"Generation {ga_instance.generations_completed} / {ga_instance.num_generations} | Best Score: {current_best_score}")


#genetic algorithm
ga_instance = pygad.GA(
    num_generations=500,
    num_parents_mating=50,
    fitness_func=fitness_func,
    sol_per_pop=100,
    num_genes=500,
    init_range_low=0,
    init_range_high=5,
    gene_type=int,
    mutation_probability=0.25,
    on_generation=on_generation,
    parallel_processing=["thread", 14]
)

print("Starting training with Bumper Guards active...")
ga_instance.run()

solution, solution_fitness, solution_idx = ga_instance.best_solution()
print(f"Training Complete! Best score: {solution_fitness}")

# --- TEST RUN ---
test_env = gym.make("Taxi-v4", render_mode="human")
state, info = test_env.reset(seed=42)

done = False
truncated = False
total_test_reward = 0
last_action = -1

# Watch the best brain drive
while not done and not truncated:
    action = int(round(solution[state]))
    action = max(0, min(5, action))

    state, reward, done, truncated, info = test_env.step(action)
    total_test_reward += reward
    time.sleep(0.2)

print(f"Final Test Run Score: {total_test_reward}")
test_env.close()