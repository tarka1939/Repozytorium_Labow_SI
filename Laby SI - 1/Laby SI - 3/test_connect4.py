import random
from copy import deepcopy

from connect4 import Connect4
from heuristics import simple_score, advanced_score
from agents import MinMaxAgent, AlphaBetaAgent


def position(rows, who_moves):
    #board from strings, top row first, e.g. '___x___'
    game = Connect4(width=len(rows[0]), height=len(rows))
    game.board = [list(row) for row in rows]
    game.who_moves = who_moves
    return game


def random_position(rng, n_moves):
    game = Connect4(width=7, height=6)
    for _ in range(n_moves):
        game.drop_token(rng.choice(game.possible_drops()))
        if game.game_over:
            break
    return game


def test_win_on_the_last_free_cell_is_not_a_tie():
    #full 4x4 board whose only four is the x column on the left
    game = position(["xoox",
                     "xxoo",
                     "xoxx",
                     "xooo"], 'o')
    assert sum(four in (list("xxxx"), list("oooo")) for four in game.iter_fours()) == 1
    assert game._check_game_over()
    assert game.wins == 'x'


def test_full_board_without_a_four_is_a_tie():
    game = position(["xoxo",
                     "xoxo",
                     "oxox",
                     "oxox"], 'x')
    assert game._check_game_over()
    assert game.wins is None


def test_simple_score_counts_fours_with_three_tokens_and_a_gap():
    game = position(["_______",
                     "_______",
                     "_______",
                     "_______",
                     "o______",
                     "xxx_oo_"], 'o')
    #x: one four (columns 0-3 of the bottom row); o: none (two tokens per four at most)
    assert simple_score(game, 'x') == 1
    assert simple_score(game, 'o') == -1


def test_advanced_score_counts_only_tokens_in_the_center_column():
    empty = Connect4(width=7, height=6)
    assert advanced_score(empty, 'x') == 0
    game = position(["_______",
                     "_______",
                     "_______",
                     "___o___",
                     "___x___",
                     "___x___"], 'o')
    assert advanced_score(game, 'x') == 2 - 1
    assert advanced_score(game, 'o') == 1 - 2


def test_alphabeta_returns_the_minmax_value():
    rng = random.Random(0)
    for heuristic in ("simple", "advanced"):
        for _ in range(10):
            game = random_position(rng, rng.randrange(0, 16))
            if game.game_over:
                continue
            minmax = MinMaxAgent(game.who_moves, depth=3, heuristic=heuristic)
            alphabeta = AlphaBetaAgent(game.who_moves, depth=3, heuristic=heuristic)
            _, value_minmax = minmax.minmax(deepcopy(game), depth=3)
            _, value_alphabeta = alphabeta.alphabeta(deepcopy(game), depth=3)
            assert value_minmax == value_alphabeta


#x to move: columns 2 and 6 win now; column 0 also wins, but only two plies later
WIN_NOW = ["_______",
           "_______",
           "_______",
           "___o___",
           "___ooo_",
           "___xxx_"]


def test_agents_take_the_immediate_win():
    for agent in (MinMaxAgent('x'), AlphaBetaAgent('x'), AlphaBetaAgent('x', heuristic="advanced")):
        assert agent.decide(position(WIN_NOW, 'x')) in (2, 6)


def test_agents_block_the_opponents_win():
    #o threatens to complete the bottom row in column 3
    rows = ["_______",
            "_______",
            "_______",
            "o______",
            "x_x____",
            "ooo_x__"]
    for agent in (MinMaxAgent('x'), AlphaBetaAgent('x'), AlphaBetaAgent('x', heuristic="advanced")):
        assert agent.decide(position(rows, 'x')) == 3


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print("ok  ", test.__name__)
    print(f"{len(tests)} tests passed")
