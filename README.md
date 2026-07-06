# Repozytorium Labów SI (AI Labs Repository)

This repository contains a series of laboratory projects completed as part of an **Artificial Intelligence / Sztuczna Inteligencja (SI)** university course. Each lab implements a core AI/ML algorithm family from scratch or with standard frameworks (NumPy, pandas, PyTorch, pygame), moving from classical search and optimization through to modern machine learning and reinforcement learning.

While written as coursework, each lab mirrors a class of problem that shows up regularly in industry. The table below summarizes what each lab does and the kind of business problem the underlying technique is used to solve in practice.

## Contents

| Lab | Topic | What it implements | Corporate / real-world value |
|-----|-------|--------------------|-------------------------------|
| [Laby SI - 2](Laby%20SI%20-%201/Laby%20SI%20-%202) | Combinatorial optimization | Brute-force solver for the 0/1 knapsack problem (`brute_force.py`, `data.py`), benchmarked on small and large datasets | Same structure as resource-allocation problems: budget/portfolio selection, cargo loading, project prioritization under a fixed capacity/cost constraint |
| [Laby SI - 3](Laby%20SI%20-%201/Laby%20SI%20-%203) | Adversarial search / game AI | Connect 4 engine with pluggable agents (random, heuristic, minimax with alpha-beta), scoring heuristics, and human-vs-agent/agent-vs-agent play modes | Foundation of automated decision-making under competition or uncertainty — used in bidding strategy, negotiation bots, and any system that must plan several moves ahead against an adversary |
| [Laby SI - 4](Laby%20SI%20-%201/Laby%20SI%20-%204) | Supervised learning | Decision Tree and Random Forest classifiers built from scratch, trained/evaluated on the Titanic survival dataset | Core technique behind churn prediction, credit scoring, fraud detection, and other classification problems where explainability matters |
| [Laby SI - 5](Laby%20SI%20-%201/Laby%20SI%20-%205) | Unsupervised learning | K-means and K-means++ clustering implemented from scratch, applied to the Iris dataset with intra-cluster variance evaluation | Directly maps to customer segmentation, market basket grouping, and anomaly/outlier detection in unlabeled data |
| [Laby SI - 6](Laby%20SI%20-%201/Laby%20SI%20-%206) | Neural networks / deep learning | A feed-forward neural network implemented manually in NumPy, then reproduced and trained with PyTorch on a two-spirals classification task | Building block for computer vision, NLP, forecasting, and any product feature powered by deep learning |
| [Laby SI - 7](Laby%20SI%20-%201/Laby%20SI%20-%207) | Reinforcement learning | Q-learning agent trained in a custom FrozenLake environment, with a pygame GUI for manual play and model save/load for train/test runs | Basis for autonomous decision systems: dynamic pricing, inventory/supply-chain control, robotics, and recommendation engines that improve from feedback |

*(`Laby SI - 1` contains the initial project scaffold used to bootstrap the later labs.)*

## Why this repository matters

Together, these labs cover the four pillars that most applied-AI work in a company draws on:

1. **Optimization** — finding the best decision under constraints (Lab 2)
2. **Search & planning** — reasoning ahead under competition or uncertainty (Lab 3)
3. **Machine learning** — learning patterns from labeled and unlabeled data (Labs 4–6)
4. **Reinforcement learning** — learning a policy through trial-and-error feedback (Lab 7)

Each implementation favors building the algorithm from first principles before using a framework, which demonstrates a working understanding of the mechanics behind the tools rather than just API usage — the kind of foundation that translates into being able to debug, tune, and adapt ML systems in production rather than treating them as black boxes.

## Structure

Each lab is a self-contained Python project (with its own `.pyproj` file for Visual Studio and a `requirements.txt`). To run a given lab:

```bash
cd "Laby SI - 1/Laby SI - <N>"
pip install -r requirements.txt
python main.py   # or the lab's entry-point script
```
