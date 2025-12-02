# Knapsack Genetic Algorithm

A flexible genetic algorithm implementation for solving the 0/1 Knapsack Problem.

## Features

- Selection methods: roulette, ranking, tournament

- Crossover: single-point, two-point

- Bit-flip mutation

- Optional repair operator for large-scale instances

- Elitism to preserve the best solution per generation

- Automatic loading of benchmark datasets with known optimal solutions

- Logs best-fitness history for analysis or plotting

## Requirements
- Python 3.6+

- No external libraries required

## Data format
**Input files:**
```
n capacity

value1 weight1

value2 weight2

...
```

**Optimum files:**

Contain a single integer representing the known optimal value.

## Quick start
```
python main.py
```
Runs a set of predefined experiments on datasets in `daneAG/` directory.


## Run a single experiment
You can run a specific dataset with custom GA parameters using `run_experiment()`:
```
run_experiment(
    data_path="daneAG/low-dimensional/f1_l-d_kp_10_269",
    selection="ranking",      # "roulette" | "ranking" | "tournament"
    crossover="single",       # "single" | "double"
    cr=0.9,                   # crossover rate  
    mr=0.05,                  # mutation rate
    pop_size=100,             # population size
    gens=500,                 # number of generations
    tournament_size=5,        # size for tournament selection
    use_repair=False          # set True for large-scale problems
)
```
## Output
- Prints best solution per generation (every 50 generations by default).

- Final summary includes best value, used weight, percentage of optimum, and runtime.

- Best-fitness history is saved automatically as:
```
hist_<dataset_name>_<selection>_<crossover>_cr<cr>_mr<mr>.txt
```
- Example outputs can be found in `tests/` directory, including tests/test.txt, which shows how the script output looks during execution.

