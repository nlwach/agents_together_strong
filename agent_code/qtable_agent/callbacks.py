import os
import pickle
import random

import numpy as np
# from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
EPSILON = 0.2
ALPHA = 0.01
GAMMA = 0.8
FEATURE_SIZE = 4


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
    self.epsilon = EPSILON  # Randomness
    self.alpha = ALPHA  # Learning rate
    self.gamma = GAMMA  # Discount factor
    self.filename = "gboost_model.pt"
    self.is_model_fit = True

    if os.path.isfile(self.filename):
        self.logger.info(f"Load existing model from {self.filename}")
        self.model = load_model(self.filename)
        self.is_model_fit = True
    elif self.train:
        self.logger.info("Setup new model")
        self.model = np.zeros((2, 2, 2, 2, len(ACTIONS)))
        # self.model = MultiOutputRegressor(GradientBoostingRegressor(learning_rate=0.6, n_estimators=20))
        # self.model = MultiOutputRegressor(MLPRegressor(hidden_layer_sizes=7, alpha=1e-5, solver="lbfgs"))
    else:
        error_message = f"Unable to find '{self.filename}'. Is the model trained yet?"
        self.logger.error(error_message)
        raise FileNotFoundError(error_message)


def load_model(filename: str):
    with open(filename, "rb") as file:
        return pickle.load(file)


def save_model(model, filename):
    with open(filename, "wb") as file:
        pickle.dump(model, file)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    if self.train and np.random.random() <= self.epsilon:
        self.logger.debug("Performing a random action.")
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])

    self.logger.debug("Querying model for action.")
    features = state_to_features(game_state)
    return ACTIONS[int(np.argmax(self.model[features]))]
    # if self.is_model_fit:
    #     prediction, = self.model.predict(features)  # Unpack prediction array
    #     action_index = np.argmax(prediction)
    #     return ACTIONS[int(action_index)]
    # else:
    #     return np.random.choice(ACTIONS, p=[.225, .225, .225, .225, .0, .1])


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

    _, _, is_bomb_ready, (player_x, player_y) = game_state["self"]
    coin_locations = game_state["coins"]
    fields = game_state["field"]
    fields[fields == -1] = 1

    player_pos = np.array([player_x, player_y])

    closest_coin_idx = np.argmin(np.linalg.norm(coin_locations - player_pos, axis=1))
    coin_x, coin_y = coin_locations[closest_coin_idx]
    coin_direction = np.array([player_x - coin_x, player_y - coin_y])
    if np.any(np.isnan(coin_direction)):
        coin_direction_norm = np.array([0, 0])
    elif np.linalg.norm(coin_direction) != 0:
        coin_direction_norm = coin_direction / np.linalg.norm(coin_direction)
    else:
        coin_direction_norm = coin_direction

    neighbouring_blocks = np.array([
        fields[player_x + 1, player_y],  # Top
        fields[player_x - 1, player_y],  # Bottom
        fields[player_x, player_y + 1],  # Left
        fields[player_x, player_y - 1],  # Right
    ], dtype=int)

    # return neighbouring_blocks.reshape((1, FEATURE_SIZE))
    return tuple(neighbouring_blocks)
