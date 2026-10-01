import random
from itertools import product

from data import get_small
from exact import knapsack_exact
from for_students import genetic_algorithm, fitness


def brute_force(weights, values, capacity):
    best = 0
    for chosen in product([False, True], repeat=len(weights)):
        weight = sum(w for w, c in zip(weights, chosen) if c)
        if weight <= capacity:
            best = max(best, sum(v for v, c in zip(values, chosen) if c))
    return best


def test_exact_matches_brute_force_on_random_instances():
    rng = random.Random(0)
    for _ in range(200):
        n = rng.randrange(0, 11)
        weights = [rng.randrange(1, 30) for _ in range(n)]
        values = [rng.randrange(1, 30) for _ in range(n)]
        capacity = rng.randrange(0, 100)
        value, chosen = knapsack_exact(weights, values, capacity)
        assert value == brute_force(weights, values, capacity)
        assert sum(weights[i] for i in chosen) <= capacity
        assert sum(values[i] for i in chosen) == value


def test_exact_on_the_small_data_set():
    items, capacity = get_small()
    value, chosen = knapsack_exact(list(items['Weight']), list(items['Value']), capacity)
    assert value == 15
    assert sorted(items['Name'][chosen]) == ['Coin', 'Crown', 'Pearls', 'Ring']


def test_genetic_algorithm_returns_a_feasible_solution():
    random.seed(0)
    items, capacity = get_small()
    solution, value, history, _ = genetic_algorithm(items, capacity, population_size=20, generations=30)
    assert value == fitness(items, capacity, solution) > 0  #fitness is 0 for overweight solutions
    assert value <= 15
    assert history == sorted(history)  #the best so far never gets worse


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print("ok  ", test.__name__)
    print(f"{len(tests)} tests passed")
