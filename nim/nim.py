import math
import random
import time
from collections import namedtuple


class Nim:
    """
    Nim game implementation – already complete.
    Do not modify this class.
    """

    def __init__(self, initial=[1, 1, 3, 5]):
        self.piles = initial.copy()
        self.player = 0
        self.winner = None

    @classmethod
    def available_actions(cls, piles):
        actions = set()
        for i, pile in enumerate(piles):
            for j in range(1, pile + 1):
                actions.add((i, j))
        return actions

    @classmethod
    def other_player(cls, player):
        return 0 if player == 1 else 1

    def switch_player(self):
        self.player = Nim.other_player(self.player)

    def move(self, action):
        pile, count = action

        if self.winner is not None:
            raise Exception("Game already won")
        if pile < 0 or pile >= len(self.piles):
            raise Exception("Invalid pile")
        if count < 1 or count > self.piles[pile]:
            raise Exception("Invalid number of objects")

        self.piles[pile] -= count
        self.switch_player()

        if all(pile == 0 for pile in self.piles):
            self.winner = self.player


class NimAI:
    """
    AI that learns to play Nim via Q‑learning.
    I implemented the four empty methods below.
    """

    def __init__(self, alpha=0.5, epsilon=0.1):
        """
        Initialize AI with an empty Q‑table,
        a learning rate alpha, and an exploration rate epsilon.
        """
        self.q = dict()
        self.alpha = alpha
        self.epsilon = epsilon

    def get_q_value(self, state, action):
        """
        Return the Q‑value for a given (state, action) pair.
        If it's never been seen before, return 0.
        """
        # state is a list; we need a tuple to use as a dict key
        state_tuple = tuple(state)
        action_tuple = tuple(action)
        return self.q.get((state_tuple, action_tuple), 0)

    def update_q_value(self, state, action, old_q, reward, future_rewards):
        """
        Update the Q‑value for (state, action) using the Q‑learning formula:
        Q(s,a) = old_q + α * (reward + future_rewards - old_q)
        """
        # New estimate = immediate reward + best future reward
        new_value = reward + future_rewards
        # Q‑learning update
        self.q[(tuple(state), tuple(action))] = old_q + self.alpha * (new_value - old_q)

    def best_future_reward(self, state):
        """
        Given a state, return the maximum Q‑value among all possible actions.
        If no actions are available, return 0.
        If an action hasn't been seen before, its Q‑value is 0.
        """
        actions = Nim.available_actions(state)
        if not actions:
            return 0

        best = max(self.get_q_value(state, action) for action in actions)
        return best

    def choose_action(self, state, epsilon=True):
        """
        Choose an action using the epsilon‑greedy algorithm.
        If epsilon is False, always pick the best (greedy) action.
        """
        actions = list(Nim.available_actions(state))

        # ε‑greedy: random move with probability epsilon
        if epsilon and random.random() < self.epsilon:
            return random.choice(actions)

        # Otherwise, pick the action with the highest Q‑value
        best_action = None
        best_q = -float('inf')  # any Q will be >= 0, so this works

        for action in actions:
            q_val = self.get_q_value(state, action)
            if q_val > best_q:
                best_q = q_val
                best_action = action

        return best_action


def train(n):
    """
    Train an AI by playing n games against itself.
    (This function is already complete – don't change it.)
    """
    ai = NimAI()

    for i in range(n):
        print(f"Playing training game {i + 1}")
        game = Nim()

        last = {
            0: {"state": None, "action": None},
            1: {"state": None, "action": None}
        }

        while True:
            state = game.piles.copy()
            action = ai.choose_action(state)

            last[game.player]["state"] = state
            last[game.player]["action"] = action

            game.move(action)
            new_state = game.piles.copy()

            if game.winner is not None:
                ai.update_q_value(state, action, ai.get_q_value(state, action), -1, 0)
                ai.update_q_value(
                    last[game.player]["state"],
                    last[game.player]["action"],
                    ai.get_q_value(last[game.player]["state"], last[game.player]["action"]),
                    0,
                    ai.best_future_reward(new_state)
                )
                ai.update_q_value(
                    new_state,
                    None,
                    ai.get_q_value(new_state, None),
                    0,
                    0
                )
                break

            elif last[game.player]["state"] is not None:
                ai.update_q_value(
                    last[game.player]["state"],
                    last[game.player]["action"],
                    ai.get_q_value(last[game.player]["state"], last[game.player]["action"]),
                    0,
                    ai.best_future_reward(new_state)
                )

    return ai


def play(ai, human_player=None):
    """
    Play a human vs. AI game.
    (This function is already complete – don't change it.)
    """
    if human_player is None:
        human_player = random.randint(0, 1)

    game = Nim()

    while True:
        print("\nPiles:")
        for i, pile in enumerate(game.piles):
            print(f"Pile {i}: {pile}")

        available_actions = Nim.available_actions(game.piles)
        print("\nAvailable actions:")
        for action in available_actions:
            pile, count = action
            print(f"Take {count} from pile {pile}")

        if game.player == human_player:
            print("\nYour Turn")
            while True:
                try:
                    pile = int(input("Choose pile: "))
                    count = int(input("Choose count: "))
                    if (pile, count) in available_actions:
                        break
                    print("Invalid action, try again.")
                except ValueError:
                    print("Invalid input, try again.")
        else:
            print("\nAI's Turn")
            action = ai.choose_action(game.piles, epsilon=False)
            pile, count = action
            print(f"AI chose to take {count} from pile {pile}.")

        game.move(action)

        if game.winner is not None:
            print("\nGame over")
            if game.winner == human_player:
                print("You win!")
            else:
                print("AI wins!")
            return
