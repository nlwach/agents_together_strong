import os
import pickle
import random
import math

import numpy as np

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Parameters for learning
    self.alpha = 0.1  # learning rate
    self.gamma = 0.6  # discount factor
    self.epsilon = 0.1  # randomness

    self.rewards = []

    if self.train:
        self.score_file = f"scores_{self.epsilon}.npy"
        if os.path.isfile(self.score_file):
            self.past_scores = np.load(self.score_file)
        else:
            self.past_scores = np.full((0, 2), np.nan)

    if self.train and not os.path.isfile(f"Q_table_{self.epsilon}.npy"):
        self.logger.info("Setting up model from scratch.")
        # if no file is present initialize Q_matrix
        self.Q_table = np.zeros((9, 6, 9, 6))
    elif self.train and os.path.isfile(f"Q_table_{self.epsilon}.npy"):
        # if training and file is present continue training old file
        self.Q_table = np.load(f"Q_table_{self.epsilon}.npy")
    else:
        # only load file no training
        self.logger.info("Loading model from saved state.")
        self.Q_table = np.load(f"Q_table_{self.epsilon}.npy")


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # Translate game into state
    state = state_to_features(game_state)

    if self.train and random.random() < self.epsilon:
        # Act randomly with a chance of self.epsilon
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])
    else:
        # Check with Q_matrix which action should be considered - choose highest reward

        action = np.argmax(self.Q_table[state[0], state[1], state[2]])
        self.logger.debug("Querying model for action.")
        # If the highest reward is 0 perform random action (state hasn't been explored)
        if self.Q_table[state[0], state[1], state[2], action] == 0:
            return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])
        # Perform aciton
        return ACTIONS[action]


def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    enemies_position = np.array(
        [game_state["others"][i][-1] for i in range(len(game_state["others"]))])

    my_position = np.array(game_state["self"][-1]) - 1

    coins = np.array(game_state["coins"]) - 1

    # Simple coord. for orientation. Agent is given the info if he is at an edge or if
    # he is at an odd or even column/row (to avoid pillars)

    upper_lower_left_right_edge = 0
    row_column_even_odd = 0

    if my_position[0] == 0:
        upper_lower_left_right_edge = 1  # lower bound
    elif my_position[0] == 14:
        upper_lower_left_right_edge = 2  # upper bound

    if my_position[1] == 14:
        if upper_lower_left_right_edge == 1:
            upper_lower_left_right_edge = 5  # upper and right bound
        elif upper_lower_left_right_edge == 2:
            upper_lower_left_right_edge = 6  # lower and right bound
        else:
            upper_lower_left_right_edge = 3  # right bound

    elif my_position[1] == 0:
        if upper_lower_left_right_edge == 1:
            upper_lower_left_right_edge = 7  # upper and left bound
        elif upper_lower_left_right_edge == 2:
            upper_lower_left_right_edge = 8  # lower and left bound
        else:
            upper_lower_left_right_edge = 4  # left bound

    if my_position[0] % 2 == 0:
        row_column_even_odd = 0  # even x
    else:
        row_column_even_odd = 1  # odd x

    if my_position[1] % 2 == 0:
        if row_column_even_odd == 0:
            row_column_even_odd = 2  # even y, even x
        elif row_column_even_odd == 1:
            row_column_even_odd = 3  # even y, odd x
    else:
        if row_column_even_odd == 0:
            row_column_even_odd = 4  # odd y, even x
        elif row_column_even_odd == 1:
            row_column_even_odd = 5  # odd y, odd x

    # Check if coins present
    if coins.size == 0:
        # last column is reserved for non coin acting
        return np.array([upper_lower_left_right_edge, row_column_even_odd, 8],
                        dtype=int)

    # Calculate closest coins angle
    closest_coin_ind = np.argmin(np.linalg.norm(coins - my_position, axis=1))
    closest_coin = coins[closest_coin_ind]

    # angle in radians
    radians = math.atan2(my_position[1] - closest_coin[1],
                         my_position[0] - closest_coin[0]) + np.pi

    # Only consider 8 directions in which coin lies
    degrees = np.degrees(radians) // 45

    return np.array([upper_lower_left_right_edge, row_column_even_odd, degrees],
                    dtype=int)
