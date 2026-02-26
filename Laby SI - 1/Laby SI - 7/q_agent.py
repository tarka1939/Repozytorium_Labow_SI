
from re import I
import numpy as np
from rl_base import Agent, Action, State
import os


class QAgent(Agent):

    def __init__(self, n_states, n_actions,
                 name='QAgent', initial_q_value=0.0, q_table=None):
        super().__init__(name)

        # hyperparams
        # TODO ustaw te parametry na sensowne wartości
        self.lr = 0.1               # współczynnik uczenia (learning rate)
        self.gamma = 0.99             # współczynnik dyskontowania
        self.epsilon = 0.9           # epsilon (p-wo akcji losowej)
        self.eps_decrement = 0.02     # wartość, o którą zmniejsza się epsilon po każdym kroku
        self.eps_min = 0.1           # końcowa wartość epsilon, poniżej którego już nie jest zmniejszane

        self.action_space = [i for i in range(n_actions)]
        self.n_states = n_states
        self.q_table = q_table if q_table is not None else self.init_q_table(initial_q_value)

    def init_q_table(self, initial_q_value=0.):
        # TODO - utwórz tablicę wartości Q o rozmiarze [n_states, n_actions] wypełnioną początkowo wartościami initial_q_value
        q_table = None
        q_table = np.full((self.n_states, len(self.action_space)), initial_q_value, dtype=np.float32)
        return q_table

    def update_action_policy(self) -> None:
        # TODO - zaktualizuj wartość epsilon

        self.epsilon = max(self.eps_min, self.epsilon - self.eps_decrement)

    def choose_action(self, state: State) -> Action:

        assert 0 <= state < self.n_states, \
            f"Bad state_idx. Has to be int between 0 and {self.n_states}"

        # TODO - zaimplementuj strategię eps-zachłanną

        if np.random.rand() < self.epsilon:
            # wybierz losową akcję
            action = np.random.choice(self.action_space)
        else:
            # wybierz akcję z największą wartością Q dla danego stanu
            action = np.argmax(self.q_table[state])
        return Action(action)
        #return Action(0)  # na razie agent zawsze wybiera akcję "idź do góry"

    def learn(self, state: State, action: Action, reward: float, new_state: State, done: bool) -> None:
        # TODO - zaktualizuj q_table
        assert 0 <= state < self.n_states, \
            f"Bad state_idx. Has to be int between 0 and {self.n_states}"
        assert 0 <= new_state < self.n_states, \
            f"Bad new_state_idx. Has to be int between 0 and {self.n_states}"
        assert 0 <= action < len(self.action_space), \
            f"Bad action_idx. Has to be int between 0 and {len(self.action_space)}"
        # Q-learning update rule
        best_next_action = np.argmax(self.q_table[new_state])
        td_target = reward + self.gamma * self.q_table[new_state][best_next_action] * (not done)
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.lr * td_error
        # TODO - zaktualizuj q_table zgodnie z regułą Q-learninga
        # Q(s, a) <- Q(s, a) + alpha * (r + gamma * max_a' Q(s', a') - Q(s, a))
        # gdzie:
        # - s to obecny stan,
        # - a to akcja podjęta w stanie s,
        # - r to nagroda otrzymana po podjęciu akcji a,
        # - s' to nowy stan po podjęciu akcji a,
        # - a' to akcje możliwe w stanie s',
        # - alpha to współczynnik uczenia (learning rate),
        # - gamma to współczynnik dyskontowania (discount factor).
        
        pass

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, self.q_table)

    def load(self, path):
        self.q_table = np.load(path)

    def get_instruction_string(self):
        return [f"Linearly decreasing eps-greedy: eps={self.epsilon:0.4f}"]

