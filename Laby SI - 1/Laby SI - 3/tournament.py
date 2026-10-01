"""Win rates and move times of the Connect Four agents (7x6 board, search depth 4).

The search agents are deterministic, so two of them would play the same game every time.
Each game therefore starts from a random opening of OPENING_MOVES moves, and every opening
is played twice, with the agents swapping colours ('o' always moves first).

usage: python tournament.py [number of openings]
"""
import random
import sys
import time
from copy import deepcopy

from connect4 import Connect4
from agents import RandomAgent, MinMaxAgent, AlphaBetaAgent

OPENINGS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
OPENING_MOVES = 2
TIMING_POSITIONS = 30
SEED = 0


def play(agents, opening):
    #agents: token -> agent; returns the winning token or None for a tie
    game = Connect4(width=7, height=6)
    for column in opening:
        game.drop_token(column)
    while not game.game_over:
        game.drop_token(agents[game.who_moves].decide(deepcopy(game)))
    return game.wins


def match(make_a, make_b, openings):
    #every opening twice, once with a as 'o' (moving first) and once as 'x'
    result = {"wins": 0, "draws": 0, "losses": 0}
    for opening in openings:
        for a_token, b_token in (('o', 'x'), ('x', 'o')):
            winner = play({a_token: make_a(a_token), b_token: make_b(b_token)}, opening)
            result["draws" if winner is None else "wins" if winner == a_token else "losses"] += 1
    return result


def random_positions(rng, count):
    positions = []
    while len(positions) < count:
        game = Connect4(width=7, height=6)
        for _ in range(rng.randrange(0, 20)):
            game.drop_token(rng.choice(game.possible_drops()))
            if game.game_over:
                break
        if not game.game_over:
            positions.append(game)
    return positions


def mean_move_time(make_agent, positions):
    start = time.perf_counter()
    for game in positions:
        make_agent(game.who_moves).decide(deepcopy(game))
    return (time.perf_counter() - start) / len(positions)


if __name__ == "__main__":
    rng = random.Random(SEED)
    random.seed(SEED)  #RandomAgent uses the global generator
    openings = [[rng.randrange(7) for _ in range(OPENING_MOVES)] for _ in range(OPENINGS)]

    matchups = [
        ("MinMax (simple)", lambda t: MinMaxAgent(t), "Random", lambda t: RandomAgent(t)),
        ("AlphaBeta (simple)", lambda t: AlphaBetaAgent(t), "Random", lambda t: RandomAgent(t)),
        ("AlphaBeta (advanced)", lambda t: AlphaBetaAgent(t, heuristic="advanced"),
         "AlphaBeta (simple)", lambda t: AlphaBetaAgent(t)),
        ("AlphaBeta (simple), depth 4", lambda t: AlphaBetaAgent(t),
         "AlphaBeta (simple), depth 2", lambda t: AlphaBetaAgent(t, depth=2)),
    ]
    print(f"{2 * OPENINGS} games per row ({OPENINGS} random openings x both colours)\n")
    print(f"| {'Agent':<30} | {'Opponent':<30} | Wins | Draws | Losses |")
    print(f"|{'-' * 32}|{'-' * 32}|------|-------|--------|")
    for name_a, make_a, name_b, make_b in matchups:
        r = match(make_a, make_b, openings)
        print(f"| {name_a:<30} | {name_b:<30} | {r['wins']:>4} | {r['draws']:>5} | {r['losses']:>6} |", flush=True)

    positions = random_positions(rng, TIMING_POSITIONS)
    print(f"\nMean time per move, same {TIMING_POSITIONS} random positions, depth 4:")
    for heuristic in ("simple", "advanced"):
        t_minmax = mean_move_time(lambda t: MinMaxAgent(t, heuristic=heuristic), positions)
        t_alphabeta = mean_move_time(lambda t: AlphaBetaAgent(t, heuristic=heuristic), positions)
        print(f"  {heuristic:<8}  MinMax {t_minmax:.3f} s   AlphaBeta {t_alphabeta:.3f} s"
              f"   ({t_alphabeta / t_minmax:.0%} of MinMax)")
