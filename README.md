# AI Lab Exercises in Python

[![CI](https://github.com/tarka1939/Repozytorium_Labow_SI/actions/workflows/ci.yml/badge.svg)](https://github.com/tarka1939/Repozytorium_Labow_SI/actions/workflows/ci.yml)

Seven lab exercises from the *Sztuczna Inteligencja* (Artificial Intelligence) course:

- linear regression
- a genetic algorithm
- minimax with alpha-beta pruning
- a random forest
- k-means
- neural networks
- Q-learning

The course supplied the scaffolding for each lab: data loaders, the Connect Four engine, the FrozenLake environment and its pygame GUI, and the plotting helpers. The algorithms were left for the student to write, and that code is mine.

After grading, every lab was checked against an exact answer or a simple baseline. That turned up nine bugs in the graded code. Three of them gave wrong results without any error:

- the random forest split on the wrong feature;
- the Connect Four heuristic scored every position 0;
- "k-means++" was actually farthest-point initialisation.

The fixes, the tests (Labs 2–5) and the experiments behind the numbers below are all in this repository. CI runs the tests, then runs every lab end to end.

## Results

| Lab | Problem | Result | Compared with |
|---|---|---|---|
| [1](#lab-1--linear-regression) | Predict fuel economy (MPG) from car weight, Auto MPG data | Test MSE **15.34**; the closed form and gradient descent give the same line | Predicting the training mean: 65.8 |
| [2](#lab-2--genetic-algorithm-for-the-knapsack-problem) | 0/1 knapsack, 26 items | With 1000 generations, finds the optimum in **17 of 20** runs (mean gap 0.04%). With 200 generations: 2 of 20 (0.26%) | Exact optimum by meet in the middle, in 0.015 s. Greedy by value/weight: 0.28% below it |
| [3](#lab-3--connect-four-minimax-and-alpha-beta) | Connect Four, search depth 4 | The positional heuristic beats the plain one **30–7–3** (win–draw–loss, 40 games). Alpha-beta needs 15–22% of minimax's time per move | Graded code: 10–20–10, because both heuristics played the same moves |
| [4](#lab-4--decision-tree-and-random-forest) | Titanic survival | Random forest: **0.811 ± 0.026** test accuracy over 20 random splits | "Women survive" rule: 0.777. Graded forest: 0.597. Majority class: 0.586 |
| [5](#lab-5--k-means) | Iris, k = 3 | k-means++ reaches the best SSE (78.94) in **91 of 100** runs | Random data points as centroids: 76 of 100. Farthest point (the graded version): 100 of 100 |
| [6](#lab-6--neural-networks) | Two interleaved spirals, 800 train / 200 test points | PyTorch MLP: **91.5%** test accuracy (95.5% train) | — |
| [7](#lab-7--q-learning) | 5×5 FrozenLake, moves succeed only 75% of the time | Q-learning policy wins **80.6–81.8%** of episodes | Random policy: 5.6% |

## Bugs found after grading

| Lab | Bug | Effect |
|---|---|---|
| 3 | `simple_score` looked for `None` as the empty cell, but empty cells are `'_'` | Every position that wasn't won or lost scored 0, so the search only saw wins and losses |
| 3 | `advanced_score` reused the name `token` as its loop variable | It added 1 for every centre cell, empty or not, so it played exactly like `simple_score` |
| 3 | A four made by the move that fills the board was reported as a draw | Wrong result in that case |
| 3 | `avp.py` imported a module that doesn't exist | Agent-vs-player mode didn't start |
| 4 | With a random feature subset, the best column was returned as an index into the subset, not the full table | Trees split on the wrong feature. On its own, this bug drops the forest from 0.81 to 0.62 |
| 4 | Bootstrap samples were drawn without replacement | Every tree saw the same data, just in a different order |
| 4 | The gain of split point *i* was computed for `y[:i]`, but the threshold sends `y[:i+1]` left | The gain didn't match the split that was applied |
| 5 | k-means++ took the farthest point instead of sampling with probability ∝ distance² | It was farthest-point initialisation under another name |
| 5 | An empty cluster's centroid moved to the origin, or the cluster was dropped | k could silently shrink |

Every bug except the `avp.py` import is covered by a test in that lab's `test_*.py` file. The commit messages give the details.

## Lab 1 — Linear regression

[`Laby SI - 1/Laby SI`](Laby%20SI%20-%201/Laby%20SI)

Predicts MPG from car weight (Auto MPG data, 392 cars, 314 train / 78 test). The coefficients come from the closed-form normal equations. Batch gradient descent on standardized weight then reaches the same line: test MSE 15.34 for both.

Changes after grading:

- The gradient descent part ended at an unfinished "TODO: calculate error"; it now reports its test error.
- The data file is in the repository; before, it was downloaded from UCI on every run.

## Lab 2 — Genetic algorithm for the knapsack problem

[`Laby SI - 1/Laby SI - 2`](Laby%20SI%20-%201/Laby%20SI%20-%202)

The GA ([`for_students.py`](Laby%20SI%20-%201/Laby%20SI%20-%202/for_students.py)) uses:

- tournament selection;
- two-point crossover;
- bit-flip mutation at rate 1/n;
- 5 elite individuals;
- a population of 100.

Overweight solutions get fitness 0.

Brute force over 2²⁶ subsets is slow, so [`exact.py`](Laby%20SI%20-%201/Laby%20SI%20-%202/exact.py) finds the optimum by **meet in the middle**:

1. List the 2¹³ subsets of each half of the items.
2. For every subset of the first half, binary-search the most valuable subset of the second half that still fits.

That takes 0.015 s and gives 13,692,887. [`test_knapsack.py`](Laby%20SI%20-%201/Laby%20SI%20-%202/test_knapsack.py) checks it against brute force on 200 random small instances.

[`experiment.py`](Laby%20SI%20-%201/Laby%20SI%20-%202/experiment.py) runs the GA with 20 seeds:

| | Runs that found the optimum | Mean gap | Worst gap | Time per run |
|---|---|---|---|---|
| GA, 200 generations (graded setting) | 2 / 20 | 0.26% | 0.71% | 0.4 s |
| GA, 1000 generations | 17 / 20 | 0.04% | 0.46% | 1.9 s |
| Greedy by value/weight | — | 0.28% | — | — |

With 200 generations the GA is no better than the greedy heuristic. It hadn't converged: in half of the runs, the final best solution first appeared at generation 129 or later.

## Lab 3 — Connect Four: minimax and alpha-beta

[`Laby SI - 1/Laby SI - 3`](Laby%20SI%20-%201/Laby%20SI%20-%203)

I wrote the minimax and alpha-beta agents ([`agents.py`](Laby%20SI%20-%201/Laby%20SI%20-%203/agents.py)) and two heuristics ([`heuristics.py`](Laby%20SI%20-%201/Laby%20SI%20-%203/heuristics.py)):

- **plain:** the player's open threes minus the opponent's, where an open three is a line of four with three of the player's tokens and one empty cell;
- **positional:** the plain score, plus 1 for each of the player's tokens in the centre column and minus 1 for each of the opponent's.

Win and loss scores depend on the search depth, so an agent takes the fastest win and the slowest loss.

The search agents are deterministic. [`tournament.py`](Laby%20SI%20-%201/Laby%20SI%20-%203/tournament.py) therefore starts each game from one of 20 random two-move openings, and plays each opening twice with the colours swapped:

| Agent | Opponent | Win–draw–loss (40 games) |
|---|---|---|
| Minimax or alpha-beta, plain heuristic | Random | 40–0–0 |
| Alpha-beta, positional heuristic | Alpha-beta, plain heuristic | 30–7–3 |
| Alpha-beta, plain heuristic, depth 4 | Alpha-beta, plain heuristic, depth 2 | 24–5–11 |

Mean time per move over 30 random positions at depth 4: minimax 0.28–0.36 s, alpha-beta 0.04–0.08 s. These are single-machine timings and vary between runs.

[`test_connect4.py`](Laby%20SI%20-%201/Laby%20SI%20-%203/test_connect4.py) has 7 tests; 4 of them fail on the graded code. One test also checks that alpha-beta returns the same value as minimax.

## Lab 4 — Decision tree and random forest

[`Laby SI - 1/Laby SI - 4`](Laby%20SI%20-%201/Laby%20SI%20-%204)

The tree, Gini split search, feature subsampling and bagging are implemented from scratch in NumPy ([`node.py`](Laby%20SI%20-%201/Laby%20SI%20-%204/node.py), [`random_forest.py`](Laby%20SI%20-%201/Laby%20SI%20-%204/random_forest.py)).

One test split has about 140 passengers, so one passenger moves accuracy by 0.7 points. That is too few to rank the models. [`experiment.py`](Laby%20SI%20-%201/Laby%20SI%20-%204/experiment.py) instead scores every model on the same 20 random 80/20 splits:

| Model | Test accuracy (mean ± std) |
|---|---|
| Random forest, 10 trees, 2 features per split | **0.811 ± 0.026** |
| "Women survive, men don't" | 0.777 ± 0.035 |
| Single decision tree, depth 14 | 0.763 ± 0.034 |
| Random forest as graded | 0.597 |
| Majority class ("everyone dies") | 0.586 ± 0.039 |

The forest beats the one-line rule on 16 of the 20 splits, but only by a few points. The rule alone closes 85% of the gap between the majority class and the forest.

The split-point fix made the gain match the split, but it didn't raise accuracy. The single tree scores 0.763 with it and 0.779 without it; the forest scores 0.811 either way.

## Lab 5 — k-means

[`Laby SI - 1/Laby SI - 5`](Laby%20SI%20-%201/Laby%20SI%20-%205)

[`experiment.py`](Laby%20SI%20-%201/Laby%20SI%20-%205/experiment.py) compares three ways to pick the initial centroids, over 100 runs each on Iris:

| Initialisation | Runs within 1% of the best SSE | Mean SSE | Mean iterations | Accuracy vs. species |
|---|---|---|---|---|
| Random data points (Forgy) | 76% | 94.5 | 6.4 | 0.802 |
| Farthest point (the graded "k-means++") | **100%** | 78.9 | 4.8 | 0.892 |
| k-means++ (correct, sampling ∝ distance²) | 91% | 84.8 | 5.7 | 0.858 |

The bug fix made the result worse on this dataset. The other runs converge to a local optimum with SSE 143–145. On Iris, the farthest-point rule happens to work well. It is known to be sensitive to outliers, but this experiment doesn't test that.

[`test_k_means.py`](Laby%20SI%20-%201/Laby%20SI%20-%205/test_k_means.py) checks:

- that the second centroid follows the distance² distribution;
- cluster assignment;
- the empty-cluster case;
- recovery of well-separated clusters.

## Lab 6 — Neural networks

[`Laby SI - 1/Laby SI - 6`](Laby%20SI%20-%201/Laby%20SI%20-%206)

- **Part 1 (NumPy):** a single neuron and a two-layer network with step activations, with the weights set by hand. The neuron classifies a linearly separable set and the network a set that isn't linearly separable, both at 100%.
- **Part 2 (PyTorch):** an MLP 2-16-32-32-16-1 with ReLU and a sigmoid output, trained on two noisy interleaved spirals. Training uses binary cross-entropy and minibatch SGD; the parameter update is written by hand rather than with `torch.optim`. The saved model ([`my_model.pkl`](Laby%20SI%20-%201/Laby%20SI%20-%206/my_model.pkl)) gets 95.5% train and 91.5% test accuracy.

The data are generated with my student number as the seed. The result comes from one training run on one split; with 200 test points, its standard error is about ±2 points.

## Lab 7 — Q-learning

[`Laby SI - 1/Laby SI - 7`](Laby%20SI%20-%201/Laby%20SI%20-%207)

A tabular Q-learning agent ([`q_agent.py`](Laby%20SI%20-%201/Laby%20SI%20-%207/q_agent.py)) with an ε-greedy policy.

The environment is a 5×5 grid with gold, two coins and three pits. A move goes in the intended direction 75% of the time, sideways 20% and backwards 5%. Episodes end at the gold, in a pit, or after 50 steps.

`python main.py test` loads the saved Q-table and plays 5000 greedy episodes. The environment isn't seeded, so over 12 such runs the win rate ranged from 80.6% to 81.8%. A random policy wins 5.6%.

ε drops by 0.02 per step, from 0.9 to its floor of 0.1, so it reaches the floor after 40 steps, within the first few episodes. Training therefore ran with an effectively constant ε = 0.1. No other schedule was compared.

## Limitations

- These are course labs on textbook datasets: 392 cars, 150 irises, about 700 Titanic passengers, and a 26-item knapsack. The value is in the implementations and in checking them, not in the scale.
- Only Labs 2–5 have unit tests. CI runs Labs 1, 6 and 7 end to end but doesn't check their numbers.
- Timings are from one machine.
- Some comments and program output are in Polish.

## Layout and running

Lab 1 is in `Laby SI - 1/Laby SI`; Labs 2–7 are in `Laby SI - 1/Laby SI - <N>`. Each lab has its own `requirements.txt`, and a Visual Studio `.pyproj`.

```bash
cd "Laby SI - 1/Laby SI - 4"
pip install -r requirements.txt
python test_random_forest.py   # tests, no pytest needed
python experiment.py           # the numbers above
```

| Lab | Main scripts |
|---|---|
| 1 | `for_students.py` |
| 2 | `for_students.py` (one GA run with plots), `exact.py`, `experiment.py`, `test_knapsack.py` |
| 3 | `ava.py` / `avp.py` / `pvp.py` (agent vs agent / agent vs player / player vs player), `tournament.py`, `test_connect4.py` |
| 4 | `main.py`, `experiment.py`, `test_random_forest.py` |
| 5 | `main.py`, `experiment.py`, `test_k_means.py` |
| 6 | `part1_net_architecture_in_numpy.py`, `part2_net_training_in_pytorch.py` (CPU PyTorch is enough) |
| 7 | `main.py` (train) or `main.py test` |

Python 3.12 in CI. Plots open in a window; set `MPLBACKEND=Agg` to run without a display.
