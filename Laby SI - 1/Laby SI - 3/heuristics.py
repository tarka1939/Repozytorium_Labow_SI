from connect4 import Connect4

"""
    Functions here should return a scalar value of a current 'position'
    in Connect4 game as seen for player playing with 'token' (one of ['o', 'x']).
"""



def simple_score(position: Connect4, token="x"):
    """Działanie:

    - Jeśli game over:
        odpowiednio 10000 ('mój token' wygrywa), -10000 ('mój token' przegrywa), 0 (w przypadku remisu)

    - Jeśli nie:
        - liczba 'czwórek' która zawiera trzy znaki typu 'mój token'
    odjąć liczba czwórek, która zawiera trzy znaki typu token przeciwnika


    Podpowiedzi:

    Skorzystać z:
        - position.iter_fours()
        - metody count() dla list: np. ['x', 'x', 'x', 'x'].count('x') -> 4

    """

    score = 0

    if position._check_game_over():
        if position.wins == token:
            return 10000
        elif position.wins is None:
            return 0
        else:
            return -10000

    opponent_token = 'o' if token == 'x' else 'x'

    for four in position.iter_fours():
        score += four.count(token) == 3 and four.count(None) == 1
        score -= four.count(opponent_token) == 3 and four.count(None) == 1

    return score

    score = 0

    raise NotImplementedError("Implement simple_score function")

    return score


def advanced_score(position: Connect4, token="x"):
    """Działanie:

    Użyj wyobraźni i stwórz własną heurystykę oceny pozycji.

    Podpowiedzi:

    - pewnie powinno być to rozwinięcie simple_score
    - metoda position.center_column() - zwraca kolumnę środkową - pewnie nie jest napisana bez powodu

    """
    score = simple_score(position, token)
    
    #add a point for each token in the center column
    #remove a point for each token of oponent in the center column
    center_column = position.center_column()
    for token in center_column:
        if token == token:
            score += 1
        elif token != None:
            score -= 1
    return score

    raise NotImplementedError("Implement advanced_score function")

    return score
