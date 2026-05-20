"""
Taxi-v4 Genetic Algorithm
"""
import gymnasium as gym
import numpy as np
import pygad
import time
import random
from collections import defaultdict

NUM_STATES  = 500
NUM_ACTIONS = 6
ZONE_COORDS = [(0, 0), (0, 4), (4, 0), (4, 3)]
OPPOSITES   = {0: 1, 1: 0, 2: 3, 3: 2}
#Fitness function
def evaluate_solution(solution, seeds):
    env = gym.make("Taxi-v4")
    total = 0.0
    for seed in seeds:
        state, info = env.reset(seed=seed)
        ep_reward   = 0.0
        done        = truncated = False
        last_action = -1
        steps       = 0
        picked_up   = False

        taxi_row, taxi_col, pass_idx, dest_idx = env.unwrapped.decode(state)
        target = ZONE_COORDS[pass_idx] if pass_idx < 4 else ZONE_COORDS[dest_idx]
        closest_dist = abs(taxi_row - target[0]) + abs(taxi_col - target[1])
        visited = {(taxi_row, taxi_col)}

        while not done and not truncated and steps < 200:
            # core lookup-table policy
            action = int(solution[state])
            action = max(0, min(5, action))

            # safety overrides (keep the taxi from getting permanently stuck)
            if action < 4 and info["action_mask"][action] == 0:
                ep_reward -= 10
                valids = [a for a in range(4) if info["action_mask"][a] == 1]
                if valids:
                    action = int(np.random.choice(valids))

            elif action < 4 and action == OPPOSITES.get(last_action, -1):
                ep_reward -= 5
                valids = [a for a in range(4)
                          if info["action_mask"][a] == 1
                          and a != OPPOSITES.get(last_action, -1)]
                if valids:
                    action = int(np.random.choice(valids))

            next_state, reward, done, truncated, info = env.step(action)
            nr, nc, np_idx, nd_idx = env.unwrapped.decode(next_state)

            # Revisit penalty
            if action < 4:
                if (nr, nc) in visited:
                    ep_reward -= 5
                else:
                    visited.add((nr, nc))

            # Distance shaping
            if not picked_up:
                target = ZONE_COORDS[np_idx] if np_idx < 4 else ZONE_COORDS[nd_idx]
            else:
                target = ZONE_COORDS[nd_idx]

            new_dist = abs(nr - target[0]) + abs(nc - target[1])
            if new_dist < closest_dist:
                ep_reward  += 3
                closest_dist = new_dist
            elif new_dist > closest_dist:
                ep_reward  -= 1

            # Pickup milestone
            if np_idx == 4 and not picked_up:
                ep_reward  += 60
                picked_up   = True
                closest_dist = abs(nr - ZONE_COORDS[nd_idx][0]) + abs(nc - ZONE_COORDS[nd_idx][1])
                visited.clear()
                visited.add((nr, nc))

            ep_reward  += reward
            last_action = action
            state       = next_state
            steps      += 1

        if done and reward == 20:
            ep_reward += 100 + max(0, 200 - steps)

        total += ep_reward

    env.close()
    return total / len(seeds)


TRAIN_SEEDS = [42, 105, 999, 7, 256, 13, 77, 512]

def fitness_func(ga_instance, solution, solution_idx):
    return evaluate_solution(solution, TRAIN_SEEDS)


# experience method
# run random rollouts: wherever a good action is observed for a state, record it.
# this gives the initial population a head start over pure random initialization.

def collect_experience(n_episodes=2000):
    env = gym.make("Taxi-v4")
    good_moves = defaultdict(list)

    for ep in range(n_episodes):
        state, info = env.reset(seed=ep)
        done = truncated = False
        trajectory = []
        picked_up  = False
        delivered  = False

        while not done and not truncated and len(trajectory) < 200:
            # move toward target, attempt pickup/dropoff when close
            taxi_row, taxi_col, pass_idx, dest_idx = env.unwrapped.decode(state)

            if not picked_up:
                target = ZONE_COORDS[pass_idx] if pass_idx < 4 else ZONE_COORDS[dest_idx]
            else:
                target = ZONE_COORDS[dest_idx]

            tr, tc = target
            dr = tr - taxi_row
            dc = tc - taxi_col

            # prefer moving toward target; pick a valid direction
            candidates = []
            if dr > 0:  candidates.append(0)   # South
            if dr < 0:  candidates.append(1)   # North
            if dc > 0:  candidates.append(2)   # East
            if dc < 0:  candidates.append(3)   # West

            # attempt pickup/dropoff when on the right cell
            if dr == 0 and dc == 0:
                if not picked_up:   candidates = [4]
                else:               candidates = [5]

            # filter to valid (non-wall) actions
            valid_candidates = [a for a in candidates if a >= 4 or info["action_mask"][a] == 1]
            if not valid_candidates:
                valid_candidates = [a for a in range(4) if info["action_mask"][a] == 1]

            action = int(np.random.choice(valid_candidates))
            trajectory.append((state, action))

            next_state, reward, done, truncated, info = env.step(action)
            _, _, np_idx, _ = env.unwrapped.decode(next_state)

            if np_idx == 4 and not picked_up:
                picked_up = True
            if done and reward == 20:
                delivered = True

            state = next_state

        # only record moves from episodes that achieved something useful
        if picked_up or delivered:
            for s, a in trajectory:
                good_moves[s].append(a)

    env.close()
    return good_moves


def build_seeded_population(pop_size, good_moves):
    # build initial population with known good moves
    population = []

    for i in range(pop_size):
        chromosome = np.random.randint(0, NUM_ACTIONS, size=NUM_STATES)

        if i < pop_size // 2 and good_moves:
            # seed this chromosome with observed good moves (with some noise)
            for state, actions in good_moves.items():
                if np.random.random() < 0.7:   # 70% chance to use the good move
                    chromosome[state] = int(np.random.choice(actions))

        population.append(chromosome)

    return population


# Island model
# Run N independent GA islands, periodically share best solutions between them.
def run_island(island_id, initial_population, generations_per_epoch):

    def _fitness(ga_inst, sol, sol_idx):
        return evaluate_solution(sol, TRAIN_SEEDS)

    def _on_gen(ga_inst):
        g = ga_inst.generations_completed
        if g % 50 == 0:
            fit = ga_inst.best_solution()[1]
            print(f"  Island {island_id} | Gen {g:>4} | Best: {fit:.1f}")

    ga = pygad.GA(
        num_generations        = generations_per_epoch,
        num_parents_mating     = max(4, len(initial_population) // 5),
        fitness_func           = _fitness,
        initial_population     = initial_population,
        num_genes              = NUM_STATES,
        gene_type              = int,
        init_range_low         = 0,
        init_range_high        = NUM_ACTIONS,
        random_mutation_min_val= 0,
        random_mutation_max_val= NUM_ACTIONS,
        mutation_probability   = 0.15,
        mutation_type          = "random",
        crossover_type         = "uniform",      #best settings for lookup tables
        parent_selection_type  = "tournament",
        keep_elitism           = 3,
        on_generation          = _on_gen,
        parallel_processing    = ["thread", 4],
        suppress_warnings      = True,
    )
    ga.run()
    return ga

#share top solutions to other islands
def migrate(islands, n_migrants=5):

    # collect best solutions from each island
    elites = []
    for ga in islands:
        pop     = ga.population
        fitness = np.array([ga.fitness_func(ga, sol, i) for i, sol in enumerate(pop)])
        top_idx = np.argsort(fitness)[-n_migrants:]
        elites.extend([pop[i].copy() for i in top_idx])

    # inject elites into each island
    for ga in islands:
        pop     = ga.population.copy()
        fitness = np.array([ga.fitness_func(ga, sol, i) for i, sol in enumerate(pop)])
        worst_idx = np.argsort(fitness)[:len(elites)]
        for rank, idx in enumerate(worst_idx):
            pop[idx] = elites[rank % len(elites)]
        ga.population = pop



#main
if __name__ == "__main__":
    N_ISLANDS       = 4
    POP_PER_ISLAND  = 100    #times n_islands=effective pop
    EPOCHS          = 3      # number of migration rounds
    GENS_PER_EPOCH  = 100    # generations each island runs between migrations

    print("="*60)
    print("Taxi-v4  —  Evolutionary Lookup Table (500 genes)")
    print(f"  Islands      : {N_ISLANDS}")
    print(f"  Pop/island   : {POP_PER_ISLAND}  (total {N_ISLANDS * POP_PER_ISLAND})")
    print(f"  Epochs       : {EPOCHS}  x  {GENS_PER_EPOCH} generations")
    print(f"  Total gens   : {EPOCHS * GENS_PER_EPOCH}")
    print("="*60)

    #Collecting exp
    print("\n[1/3] Collecting experience via heuristic rollouts...")
    good_moves = collect_experience(n_episodes=3000)
    print(f"      Observed good moves for {len(good_moves)} / {NUM_STATES} states")

    #building seeded pop
    print("\n[2/3] Building seeded initial populations...")
    island_populations = [
        build_seeded_population(POP_PER_ISLAND, good_moves)
        for _ in range(N_ISLANDS)
    ]

    #island evolution with migration
    print("\n[3/3] Running island model GA...\n")
    islands = []
    for epoch in range(EPOCHS):
        print(f"── Epoch {epoch + 1} / {EPOCHS} ──────────────────────────────")

        if epoch == 0:
            # first epoch: initialize all islands
            islands = [
                run_island(i, np.array(island_populations[i]), GENS_PER_EPOCH)
                for i in range(N_ISLANDS)
            ]
        else:
            # subsequent epochs: continue from current populations
            islands = [
                run_island(i, islands[i].population, GENS_PER_EPOCH)
                for i in range(N_ISLANDS)
            ]

        if epoch < EPOCHS - 1:
            print(f"\n  → Migrating top solutions between islands...\n")
            migrate(islands, n_migrants=5)

    # pickup best solution across islands
    best_solution  = None
    best_fitness   = -1e9

    for ga in islands:
        sol, fit, _ = ga.best_solution()
        if fit > best_fitness:
            best_fitness  = fit
            best_solution = sol.copy()

    print(f"\nTraining complete! Best avg fitness: {best_fitness:.2f}")
    np.save("best_taxi_policy.npy", best_solution)
    print("Policy saved to best_taxi_policy.npy")

    # test run
    print("\n--- Test Run (seed=42) ---")
    test_env = gym.make("Taxi-v4", render_mode="human")
    state, info = test_env.reset(seed=42)

    done = truncated = False
    total_reward = 0
    steps = 0
    last_action = -1

    while not done and not truncated and steps < 200:
        action = int(best_solution[state])
        action = max(0, min(5, action))

        if action < 4 and info["action_mask"][action] == 0:
            valids = [a for a in range(4) if info["action_mask"][a] == 1]
            if valids:
                action = int(np.random.choice(valids))

        state, reward, done, truncated, info = test_env.step(action)
        total_reward += reward
        last_action   = action
        steps        += 1
        time.sleep(0.15)

    print(f"Test reward: {total_reward}  |  Steps: {steps}")
    test_env.close()