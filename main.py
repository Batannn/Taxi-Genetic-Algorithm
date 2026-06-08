
import gymnasium as gym
import numpy as np
import pygad
import time
import csv
from collections import defaultdict

NUM_STATES  = 500
NUM_ACTIONS = 6
ZONE_COORDS = [(0, 0), (0, 4), (4, 0), (4, 3)]
OPPOSITES   = {0: 1, 1: 0, 2: 3, 3: 2}
FITNESS_LOG = "fitness_log.csv"

def init_log():
    with open(FITNESS_LOG, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "island", "gen", "best_fitness"])

def append_log(epoch, island_id, gen, fitness):
    with open(FITNESS_LOG, "a", newline="") as f:
        csv.writer(f).writerow([epoch, island_id, gen, round(fitness, 2)])


CURRENT_EPOCH = 1


def get_curriculum(epoch):
    """Return penalty/reward magnitudes based on current training phase."""
    if epoch <= 2:
        return dict(
            wall_pen=0,  # no punishment for hitting walls
            revisit_pen=0,  # no punishment for circling
            backtrack_pen=0,  # no punishment for going back
            closer_bonus=2,  # tiny nudge toward target
            farther_pen=0,  # no punishment for moving away
            pickup_bonus=200,  # HUGE reward for finding passenger
            dropoff_bonus=500,  # HUGE reward for delivery
            speed_bonus=0,  # no speed pressure yet
        )
    elif epoch <= 4:
        return dict(
            wall_pen=-5,
            revisit_pen=-3,
            backtrack_pen=-3,
            closer_bonus=3,
            farther_pen=-1,
            pickup_bonus=150,
            dropoff_bonus=400,
            speed_bonus=0,
        )
    else:
        return dict(
            wall_pen=-10,
            revisit_pen=-5,
            backtrack_pen=-5,
            closer_bonus=3,
            farther_pen=-1,
            pickup_bonus=100,
            dropoff_bonus=300,
            speed_bonus=100,  # bonus for finishing fast
        )
#Episode runner
def run_episode(solution, env, seed, epoch, already_picked_up=False):
    c = get_curriculum(epoch)
    state, info = env.reset(seed=seed)
    taxi_row, taxi_col, pass_idx, dest_idx = env.unwrapped.decode(state)

    if already_picked_up:
        forced_state = env.unwrapped.encode(taxi_row, taxi_col, 4, dest_idx)
        env.unwrapped.s = forced_state
        state = forced_state
        _, info = env.reset(seed=seed)
        env.unwrapped.s = forced_state
        state = forced_state
        info["action_mask"] = env.unwrapped.action_mask(forced_state)

    ep_reward   = 0.0
    raw_reward  = 0.0
    done        = truncated = False
    last_action = -1
    steps       = 0
    picked_up   = already_picked_up

    taxi_row, taxi_col, pass_idx, dest_idx = env.unwrapped.decode(state)
    target = ZONE_COORDS[dest_idx] if picked_up else (
        ZONE_COORDS[pass_idx] if pass_idx < 4 else ZONE_COORDS[dest_idx]
    )
    closest_dist = abs(taxi_row - target[0]) + abs(taxi_col - target[1])
    visited = {(taxi_row, taxi_col)}

    while not done and not truncated and steps < 200:
        action = int(solution[state])
        action = max(0, min(5, action))

        if action < 4 and info["action_mask"][action] == 0:
            ep_reward +=c["wall_pen"]
            valids = [a for a in range(4) if info["action_mask"][a] == 1]
            if valids:
                action = int(np.random.choice(valids))
        elif action < 4 and action == OPPOSITES.get(last_action, -1):
            ep_reward += c["backtrack_pen"]
            valids = [a for a in range(4)
                      if info["action_mask"][a] == 1
                      and a != OPPOSITES.get(last_action, -1)]
            if valids:
                action = int(np.random.choice(valids))

        next_state, reward, done, truncated, info = env.step(action)
        nr, nc, np_idx, nd_idx = env.unwrapped.decode(next_state)

        if action < 4:
            if (nr, nc) in visited:
                ep_reward += c["revisit_pen"]
            else:
                visited.add((nr, nc))

        if not picked_up:
            target = ZONE_COORDS[np_idx] if np_idx < 4 else ZONE_COORDS[nd_idx]
        else:
            target = ZONE_COORDS[nd_idx]

        new_dist = abs(nr - target[0]) + abs(nc - target[1])
        if new_dist < closest_dist:
            ep_reward   += c["closer_bonus"]
            closest_dist = new_dist
        elif new_dist > closest_dist:
            ep_reward   +=c["farther_pen"]

        if np_idx == 4 and not picked_up:
            ep_reward   += c["pickup_bonus"]
            picked_up    = True
            closest_dist = abs(nr - ZONE_COORDS[nd_idx][0]) + abs(nc - ZONE_COORDS[nd_idx][1])
            visited.clear()
            visited.add((nr, nc))

        ep_reward  += reward
        raw_reward += reward
        last_action = action
        state       = next_state
        steps      += 1

    if done and reward == 20:
        ep_reward += c["dropoff_bonus"] + c["speed_bonus"] * max(0, 200 - steps) / 200

    return ep_reward,raw_reward, steps, picked_up, (done and reward == 20)


# Seeds are weighted by difficulty — harder seeds count more.
# Weights are updated automatically after each epoch based on pass/fail history.

TRAIN_SEEDS   = [42, 105, 999, 7, 256, 13, 77, 512]
#Start
SEED_WEIGHTS  = {s: 1.0 for s in TRAIN_SEEDS}

def evaluate_solution(solution, seeds, epoch):
    env   = gym.make("Taxi-v4")
    total = 0.0
    weight_sum = 0.0

    for seed in seeds:
        w = SEED_WEIGHTS[seed]
        score_a, _, _, _, _ = run_episode(solution, env, seed, epoch, already_picked_up=False)
        score_b, _, _, _, _ = run_episode(solution, env, seed, epoch, already_picked_up=True)
        total      += w * (score_a + 0.4 * score_b)
        weight_sum += w

    env.close()
    return total / weight_sum


def fitness_func(ga_instance, solution, solution_idx):
    return evaluate_solution(solution, TRAIN_SEEDS, CURRENT_EPOCH)


def update_seed_weights(best_solution):
    """
    Re-evaluate the best solution on all seeds and increase weight for seeds
    where it fails (no pickup). This forces the GA to care about hard seeds.
    """
    env = gym.make("Taxi-v4")
    for seed in TRAIN_SEEDS:
        _, _, _, picked_up, delivered = run_episode(best_solution, env, seed, CURRENT_EPOCH)
        if not picked_up:
            SEED_WEIGHTS[seed] = min(SEED_WEIGHTS[seed] * 1.5, 6.0)  # cap at 6x
        elif not delivered:
            SEED_WEIGHTS[seed] = min(SEED_WEIGHTS[seed] * 1.2, 6.0)
        else:
            # It already handles this seed — reduce weight so others matter more
            SEED_WEIGHTS[seed] = max(SEED_WEIGHTS[seed] * 0.8, 0.5)
    env.close()
    w_str = {s: f"{w:.1f}x" for s, w in SEED_WEIGHTS.items()}
    print(f"  Seed weights updated: {w_str}")


#Diversity check
def inject_diversity(ga, fraction=0.4):

    cached = ga.last_generation_fitness
    if cached is None:
        return False

    top5_spread = np.max(cached) - np.sort(cached)[-5]
    if top5_spread > 1.0:
        return False   # still diverse enough

    pop       = ga.population.copy()
    n_replace = int(len(pop) * fraction)
    worst_idx = np.argsort(cached)[:n_replace]

    for idx in worst_idx:
        pop[idx] = np.random.randint(0, NUM_ACTIONS, size=NUM_STATES)

    ga.population = pop
    return True   # diversity was injected


#Terminal quick-test

def terminal_test(solution, epoch):
    env = gym.make("Taxi-v4")
    print(f"\n  ┌─ Epoch {epoch} quick-test ───────────────────────────────")
    print(f"  │  {'seed':<6}  {'raw score':>10}  {'steps':<6}  result")
    print(f"  │  {'':─<6}  {'':─>10}  {'':─<6}  {'':─<12}")
    pickups = deliveries = 0
    for seed in [42, 105, 999, 7, 256]:
        _, raw, steps, picked_up, delivered = run_episode(solution, env, seed, epoch)
        w      = SEED_WEIGHTS[seed]
        status = "✓ DELIVERED" if delivered else ("↑ PICKED UP" if picked_up else "✗ no pickup")
        print(f"  │  {seed:<6}  {raw:>10.1f}  {steps:<6}  {status}")
        if picked_up:   pickups    += 1
        if delivered:   deliveries += 1
    print(f"  │  Pickups: {pickups}/5   Deliveries: {deliveries}/5")
    print(f"  └───────────────────────────────────────────────────────")
    env.close()


#Experience collection

def collect_experience(n_episodes=3000):
    env        = gym.make("Taxi-v4")
    good_moves = defaultdict(list)

    for ep in range(n_episodes):
        state, info = env.reset(seed=ep)
        done = truncated = False
        trajectory = []
        picked_up  = False

        while not done and not truncated and len(trajectory) < 200:
            taxi_row, taxi_col, pass_idx, dest_idx = env.unwrapped.decode(state)
            target = ZONE_COORDS[dest_idx] if picked_up else (
                ZONE_COORDS[pass_idx] if pass_idx < 4 else ZONE_COORDS[dest_idx]
            )
            tr, tc  = target
            dr, dc  = tr - taxi_row, tc - taxi_col
            candidates = []
            if dr > 0: candidates.append(0)
            if dr < 0: candidates.append(1)
            if dc > 0: candidates.append(2)
            if dc < 0: candidates.append(3)
            if dr == 0 and dc == 0:
                candidates = [4] if not picked_up else [5]

            valid_candidates = [a for a in candidates if a >= 4 or info["action_mask"][a] == 1]
            if not valid_candidates:
                valid_candidates = [a for a in range(4) if info["action_mask"][a] == 1]

            action = int(np.random.choice(valid_candidates))
            trajectory.append((state, action))
            next_state, reward, done, truncated, info = env.step(action)
            _, _, np_idx, _ = env.unwrapped.decode(next_state)
            if np_idx == 4 and not picked_up: picked_up = True
            state = next_state

        if picked_up:
            for s, a in trajectory:
                good_moves[s].append(a)

    env.close()
    return good_moves


def build_seeded_population(pop_size, good_moves):
    population = []
    for i in range(pop_size):
        chromosome = np.random.randint(0, NUM_ACTIONS, size=NUM_STATES)
        if i < pop_size // 2 and good_moves:
            for state, actions in good_moves.items():
                if np.random.random() < 0.7:
                    chromosome[state] = int(np.random.choice(actions))
        population.append(chromosome)
    return population


def run_island(island_id, initial_population, generations_per_epoch, epoch):
    log_buffer = []

    def _fitness(ga_inst, sol, sol_idx):
        return evaluate_solution(sol, TRAIN_SEEDS, epoch)

    def _on_gen(ga_inst):
        g   = ga_inst.generations_completed
        fit = ga_inst.best_solution()[1]
        log_buffer.append((epoch, island_id, g, fit))
        if g % 50 == 0:
            print(f"  Island {island_id} | Gen {g:>4} | Best: {fit:.1f}")

    ga = pygad.GA(
        num_generations         = generations_per_epoch,
        num_parents_mating      = max(4, len(initial_population) // 4),
        fitness_func            = _fitness,
        initial_population      = initial_population,
        num_genes               = NUM_STATES,
        gene_type               = int,
        init_range_low          = 0,
        init_range_high         = NUM_ACTIONS,
        random_mutation_min_val = 0,
        random_mutation_max_val = NUM_ACTIONS,
        mutation_probability    = 0.25,
        mutation_type           = "random",
        crossover_type          = "uniform",
        parent_selection_type   = "rws",
        keep_elitism            = 1,
        on_generation           = _on_gen,
        parallel_processing     = ["thread", 4],
        suppress_warnings       = True,
    )
    ga.run()

    for row in log_buffer:
        append_log(*row)

    return ga


def migrate(islands, n_migrants=2):
    elites = []
    for ga in islands:
        sol, fit, _ = ga.best_solution()
        elites.append((fit, sol.copy()))
    elites.sort(key=lambda x: x[0], reverse=True)
    elite_chromosomes = [sol for _, sol in elites]

    for ga in islands:
        pop           = ga.population.copy()
        cached        = ga.last_generation_fitness
        if cached is None or len(cached) != len(pop):
            worst_idx = list(range(min(n_migrants, len(pop))))
        else:
            worst_idx = np.argsort(cached)[:n_migrants].tolist()
        for rank, idx in enumerate(worst_idx):
            pop[idx] = elite_chromosomes[rank % len(elite_chromosomes)].copy()
        ga.population = pop


#Main

if __name__ == "__main__":

    N_ISLANDS      = 4
    POP_PER_ISLAND = 100
    EPOCHS         = 5
    GENS_PER_EPOCH = 100
    print("=" * 60)
    print("Taxi-v4  —  Evolutionary Lookup Table (500 genes) v5")
    print(f"  Islands      : {N_ISLANDS}")
    print(f"  Pop/island   : {POP_PER_ISLAND}  (total {N_ISLANDS * POP_PER_ISLAND})")
    print(f"  Epochs       : {EPOCHS}  x  {GENS_PER_EPOCH} generations")
    print(f"  Fitness log  : {FITNESS_LOG}")
    print(f"  Epochs 1-2 : free exploration, no penalties")
    print(f"  Epochs 3-4 : mild guidance introduced")
    print(f"  Epochs 5+  : full penalties + speed pressure")
    print("=" * 60)

    init_log()

    print("\n[1/3] Collecting experience via heuristic rollouts...")
    good_moves = collect_experience(n_episodes=3000)
    print(f"      Observed good moves for {len(good_moves)} / {NUM_STATES} states")

    print("\n[2/3] Building seeded initial populations...")
    island_populations = [
        build_seeded_population(POP_PER_ISLAND, good_moves)
        for _ in range(N_ISLANDS)
    ]

    print("\n[3/3] Running island model GA...\n")
    best_ever     = -1e9
    best_ever_raw = -1e9
    best_deliveries = 0
    best_solution = None
    islands       = []

    for epoch in range(1, EPOCHS + 1):
        CURRENT_EPOCH = epoch
        print(f"── Epoch {epoch} / {EPOCHS} ──────────────────────────────")

        if epoch == 1:
            islands = [
                run_island(i, np.array(island_populations[i]), GENS_PER_EPOCH, epoch)
                for i in range(N_ISLANDS)
            ]
        else:
            # Diversity check before running — inject fresh blood if converged
            for i, ga in enumerate(islands):
                if inject_diversity(ga):
                    print(f"  ⚡ Island {i}: diversity injected (population had converged)")
            islands = [
                run_island(i, islands[i].population, GENS_PER_EPOCH, epoch)
                for i in range(N_ISLANDS)
            ]

        # Best this epoch
        epoch_best_sol = None
        epoch_best_fit = -1e9
        for ga in islands:
            sol, fit, _ = ga.best_solution()
            if fit > epoch_best_fit:
                epoch_best_fit = fit
                epoch_best_sol = sol.copy()

        env_check = gym.make("Taxi-v4")
        epoch_deliveries = 0
        epoch_raw_total = 0.0
        for seed in TRAIN_SEEDS:
            _, raw, _, _, delivered = run_episode(epoch_best_sol, env_check, seed, epoch)
            epoch_raw_total += raw
            if delivered: epoch_deliveries += 1
        env_check.close()
        epoch_raw_avg = epoch_raw_total / len(TRAIN_SEEDS)

        # Save if better deliveries, or same deliveries but better raw score
        if (epoch_deliveries > best_deliveries or
                (epoch_deliveries == best_deliveries and epoch_raw_avg > best_ever_raw)):
            best_deliveries = epoch_deliveries
            best_ever_raw = epoch_raw_avg
            best_ever = epoch_best_fit
            best_solution = epoch_best_sol.copy()
            np.save("best_taxi_policy.npy", best_solution)

        terminal_test(epoch_best_sol, epoch)
        print(f"  Best so far: {best_deliveries}/8 seeds delivered  |  raw avg: {best_ever_raw:.1f}\n")

        # Migrate only on even epochs to avoid premature homogenization
        if epoch < EPOCHS and epoch % 2 == 0:
            print(f"  → Migrating top solutions between islands...\n")
            migrate(islands, n_migrants=2)

    print(f"\nTraining complete! Best avg fitness: {best_ever:.2f}")
    print(f"Policy saved to best_taxi_policy.npy")
    print(f"Fitness log saved to {FITNESS_LOG}")

    #Test run: BROKEN
    print("\n--- Full Test Run with GUI (seed=42) ---")
    test_env = gym.make("Taxi-v4", render_mode="human")
    state, info = test_env.reset(seed=42)
    done = truncated = False
    total_reward = 0
    steps = 0

    while not done and not truncated and steps < 200:
        action = int(best_solution[state])
        action = max(0, min(5, action))
        if action < 4 and info["action_mask"][action] == 0:
            valids = [a for a in range(4) if info["action_mask"][a] == 1]
            if valids:
                action = int(np.random.choice(valids))
        state, reward, done, truncated, info = test_env.step(action)
        total_reward += reward
        steps        += 1
        time.sleep(0.15)

    print(f"Test reward: {total_reward}  |  Steps: {steps}")
    test_env.close()