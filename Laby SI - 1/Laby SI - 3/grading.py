from agents import MinMaxAgent, AlphaBetaAgent, RandomAgent
from ava import compare_AIs


def print_info(points_to_get, condition):
    suffix = "s" if points_to_get > 1 else ""
    if condition:
        print(f"OK. Enough for {points_to_get} point{suffix}! :)")
    else:
        print(f"Not enough for {points_to_get} point{suffix}. :(")


def for_1_point():
    n_games = 5
    agent1 = RandomAgent("o")
    agent2 = MinMaxAgent("x")
    results, times = compare_AIs(agent1, agent2, n_games)
    print_info(1, results["x"] >= 0.8 * n_games)


def for_2_points():
    n_games = 5
    agent1 = RandomAgent("o")
    agent2 = MinMaxAgent("x")
    _, times1 = compare_AIs(agent1, agent2, n_games)
    agent1 = RandomAgent("o")
    agent2 = AlphaBetaAgent("x")
    results, times2 = compare_AIs(agent1, agent2, n_games)
    print_info(
        2,
        results["x"] >= 0.8 * n_games and times2["x"] < 0.25 * times1["x"],
    )


def for_3_points():
    n_games = 5
    agent1 = AlphaBetaAgent("o", heuristic="simple")
    agent2 = AlphaBetaAgent("x", heuristic="advanced")
    results, times = compare_AIs(agent1, agent2, n_games)
    print_info(3, results["x"] >= 0.8 * n_games)


if __name__ == "__main__":
    #for_1_point()
    #print("stop")
    for_2_points()
    #for_3_points()
