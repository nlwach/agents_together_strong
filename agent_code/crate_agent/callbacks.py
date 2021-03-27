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
    #Parameters for learning
    self.alpha = 0.1 #learning rate
    self.gamma = 0.6 #discount factor
    self.epsilon = 0.25 #randomness
    if self.train and not os.path.isfile(f"Q_table_{self.epsilon}.npy"):
        self.logger.info("Setting up model from scratch.")

        #if no file is present initialize Q_matrix
        self.Q_table = np.zeros((15, 16, 5, 5, 6))
        self.model = None

    elif self.train and os.path.isfile(f"Q_table_{self.epsilon}.npy"):
        #if training and file is present continue training old file
        self.Q_table = np.load(f"Q_table_{self.epsilon}.npy")

    else:
        #only load file no training
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
    #Translate game into state
    state = state_to_features(game_state)

    if self.train and random.random() < self.epsilon:
        #Act randomly with a chance of self.epsilon
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])
    else:
        #Check with Q_matrix which action should be considered - choose highest reward
        #print(state, self.Q_table.shape)
        action = np.argmax(self.Q_table[tuple(state)])
        self.logger.debug("Querying model for action.")
        #If the highest reward is 0 perform random action (state hasn't been explored)
        #print(state, state.shape)
        if self.Q_table[state[0], state[1], state[2], state[3], action] == 0:
            return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])
        #Perform aciton
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



    enemies_position = np.array([game_state["others"][i][-1] for i in range(len(game_state["others"]))])

    my_position = np.array(game_state["self"][-1])

    coins = np.array(game_state["coins"])

    field = np.array(game_state["field"])

    bombs = game_state["bombs"]
    bomb_pos = np.array([tup[0] for tup in bombs])
    bomb_timers = np.array([tup[1] for tup in bombs])
    explosion_map = np.asarray(game_state["explosion_map"])
    bomb_rel_to_pos = [4, 4]
    closest_timer = 5

    if bomb_pos != []:
        bomb_dist = np.linalg.norm(my_position - bomb_pos)
        closest_bombs = np.argsort(bomb_dist)

        closest_bomb = bomb_pos[closest_bombs[0]]
        if np.linalg.norm(closest_bomb - my_position)<3:
            closest_timer = bomb_timers[closest_bombs[0]]

            bomb_rel_to_pos = closest_bomb - my_position

    top_is_blocked = 0
    bottom_is_blocked = 0
    left_is_blocked = 0
    right_is_blocked = 0

    is_crate_left = 0
    is_crate_right = 0
    is_crate_bottom = 0
    is_crate_top = 0

    explosion_left = 0
    explosion_right = 0
    explosion_top = 0
    explosion_bottom = 0


    if field[my_position[0]+1, my_position[1]]:
        right_is_blocked = 1
        if field[my_position[0]+1, my_position[1]] == 1:
            is_crate_right = 1
    if field[my_position[0]-1, my_position[1]]:
        left_is_blocked = 1
        if field[my_position[0]-1, my_position[1]] == 1:
            is_crate_left = 1
    if field[my_position[0], my_position[1]-1]:
        bottom_is_blocked = 1
        if field[my_position[0], my_position[1]-1] == 1:
            is_crate_bottom = 1
    if field[my_position[0], my_position[1]+1]:
        if field[my_position[0], my_position[1]+1] == 1:
            is_crate_top = 1
        top_is_blocked = 1



    if explosion_map[my_position[0] + 1, my_position[1]]:
        explosion_right = 1
    if explosion_map[my_position[0] - 1, my_position[1]]:
        explosion_left = 1
    if explosion_map[my_position[0], my_position[1] - 1]:
        explosion_bottom = 1
    if explosion_map[my_position[0], my_position[1] + 1]:
        explosion_top = 1

    crate_list = [is_crate_right, is_crate_left, is_crate_bottom, is_crate_top]
    explosion_list = [explosion_right, explosion_left, explosion_bottom, explosion_top]
    blocked_list = [right_is_blocked, left_is_blocked, bottom_is_blocked, top_is_blocked]



    crate_int = int("".join(str(i) for i in crate_list), 2)
    blocked_int = int("".join(str(i) for i in blocked_list), 2)
    explosion_int = int("".join(str(i) for i in explosion_list), 2)


#return np.array([crate_int, blocked_int, explosion_int, bomb_rel_to_pos[0], bomb_rel_to_pos[1], closest_timer], dtype=int)
    return np.array([blocked_int, explosion_int, bomb_rel_to_pos[0], bomb_rel_to_pos[1]],
                    dtype=int)
"""
    if coins.size == 0:
        #last column is reserved for non coin acting
        return np.array([crate_int, blocked_int, explosion_int])

    #Calculate closest coins angle
    closest_coin_ind = np.argmin(np.linalg.norm(coins-my_position, axis=1))
    closest_coin = coins[closest_coin_ind]

    #angle in radians
    radians = math.atan2(my_position[1]-closest_coin[1], my_position[0]-closest_coin[0])+np.pi

    #Only consider 8 directions in which coin lies
    degrees = np.degrees(radians)//45
"""



