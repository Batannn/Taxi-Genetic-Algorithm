# Taxi Route Optimization using Genetic Algorithm

**Biologically Inspired Artificial Intelligence - University Project**

## Overview
This project implements a Genetic Algorithm (GA) to solve the classic Reinforcement Learning "Taxi" environment (`Taxi-v4`) from OpenAI's Gymnasium. The goal of the project is to train an AI agent to navigate a grid, successfully pick up a passenger, and drop them off at their destination by evolving an optimal set of instructions over multiple generations.

## How it Works
Instead of using a traditional Neural Network or Q-Learning, this project uses a **Direct Policy Search** via an Evolutionary Lookup Table.
- **State Space:** The Taxi environment has exactly 500 discrete states (representing taxi location, passenger location, and destination).
- **Action Space:** The agent can take 6 possible actions (Move South, North, East, West, Pickup, Dropoff).
- **The Chromosome:** Each potential solution in our genetic population is represented as an array of 500 integers (ranging from 0 to 5). The index represents the state, and the value represents the action.
- **Evolution:** Using **PyGAD**, we evaluate the fitness of these policy arrays based on the total reward accumulated during an episode. The best-performing policies are selected, crossed over, and mutated to create increasingly efficient generations.

## Tech Stack
* **Python 3**
* **PyGAD** - Handles the Genetic Algorithm mechanics (Selection, Crossover, Mutation).
* **Gymnasium** - Provides the `Taxi-v4` simulation environment.
* **Matplotlib** - Used to visualize the fitness improvements across generations.

## Installation & Setup

1. **Clone the repository:**
   git clone https://github.com/Batannn/Taxi-Genetic-Algorithm.git
   cd Taxi-Genetic-Algorithm

2. **Set up the environment:**
   If you are using PyCharm, open the cloned folder and let PyCharm automatically create the virtual environment. Then, open the terminal at the bottom and run:
   pip install gymnasium[toy-text] pygad matplotlib

## Usage
*(Instructions on how to run the training script will be added here once development is complete)*
python main.py

## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contributors
* **Batannn**
* **kajaszanska**
