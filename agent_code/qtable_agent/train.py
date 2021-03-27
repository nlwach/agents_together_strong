import os
from collections import namedtuple, deque
from typing import List

import events as e
from .callbacks import state_to_features, save_model, FEATURE_SIZE

import numpy as np

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 20  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
CLOSER_TO_COIN = "CLOSER_TO_COIN"
FURTHER_FROM_COIN = "FURTHER_FROM_COIN"

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.rewards = []
    self.score_file = f"scores.npy"
    if os.path.isfile(self.score_file):
        self.past_scores = np.load(self.score_file)
    else:
        self.past_scores = np.full((0, 2), np.nan)
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)


def game_events_occurred(self, old_game_state: dict, self_action: str,
                         new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(
        f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # Idea: Add your own events to hand out rewards
    old_features = state_to_features(old_game_state)
    new_features = state_to_features(new_game_state)

    reward_coin_distance(old_game_state, new_game_state, events)
    reward = reward_from_events(self, events)
    self.rewards.append(reward)

    # state_to_features is defined in callbacks.py
    self.transitions.append(Transition(old_features, self_action, new_features, reward))
    if len(self.transitions) == self.transitions.maxlen:
        train_batch(self, self.transitions)


def train_batch(self, batch: deque):
    batch_size: int = len(batch)
    if batch_size < 1:
        return
    self.logger.info(f"Training batch of {batch_size} transitions")
    # X = np.full((batch_size, FEATURE_SIZE), np.nan)
    # y = np.full((batch_size, len(ACTIONS)), np.nan)
    for i, (old_features, action, new_features, reward) in enumerate(batch):
        if old_features is not None and action is not None:
            action_index = ACTIONS.index(action)
            if not self.is_model_fit:
                old_value = 0
                next_value = 0
                q_values = np.zeros(len(ACTIONS))
            else:
                # q_values = self.model.predict(old_features)[0]
                q_values = self.model[old_features]
                old_value = q_values[action_index]
                if new_features is not None:
                    # next_value = np.max(self.model.predict(new_features))
                    next_value = np.max(self.model[new_features])
                else:
                    next_value = 0
            q_value_updated = (
                    (1 - self.alpha) * old_value
                    + self.alpha * (reward + self.gamma * next_value)
            )
            q_values[action_index] = q_value_updated
            # y[i] = q_values
            # X[i] = old_features
            self.model[old_features] = q_values
    batch.clear()
    # y_nan = np.any(np.isnan(y), axis=1)
    # if len(y[~y_nan]) == 0:
    #     return
    # self.model.fit(X[~y_nan], y[~y_nan])
    # self.is_model_fit = True


def reward_coin_distance(old_game_state, new_game_state, events):
    if old_game_state is None or new_game_state is None:
        return
    old_coins = np.array(old_game_state["coins"]) - 1
    new_coins = np.array(new_game_state["coins"]) - 1

    # Check if coin was present and is still present
    if old_coins != [] and new_coins != []:
        old_pos = np.array(old_game_state["self"][-1]) - 1
        new_pos = np.array(new_game_state["self"][-1]) - 1

        old_distance = np.linalg.norm(old_coins - old_pos, axis=1)
        new_distance = np.linalg.norm(new_coins - new_pos, axis=1)

        index_old_coin = np.argmin(np.linalg.norm(old_coins - old_pos, axis=1))
        index_new_coin = np.argmin(np.linalg.norm(new_coins - old_pos, axis=1))

        # Check if it is the same coin
        if np.all(old_coins[index_old_coin] == new_coins[index_new_coin]):
            if np.min(old_distance) > np.min(new_distance):
                events.append(CLOSER_TO_COIN)
            elif np.min(old_distance) < np.min(new_distance):
                events.append(FURTHER_FROM_COIN)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.

    This is similar to reward_update. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(
        f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    self.transitions.append(
        Transition(state_to_features(last_game_state), last_action, None,
                   reward_from_events(self, events)))

    train_batch(self, self.transitions)

    mean_reward = np.mean(self.rewards)
    median_reward = np.median(self.rewards)
    reward_std = np.std(self.rewards)
    reward_sum = np.sum(self.rewards)
    if self.past_scores.size == 0:
        self.past_scores = np.array([
            [mean_reward, median_reward, reward_std, reward_sum]
        ])
    else:
        self.past_scores = np.append(
            self.past_scores,
            np.array([[mean_reward, median_reward, reward_std, reward_sum]]),
            axis=0
        )
    np.save(self.score_file, self.past_scores)
    self.rewards = []
    save_model(self.model, self.filename)


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        # e.COIN_COLLECTED: 20,
        # e.KILLED_OPPONENT: 30,
        e.KILLED_SELF: -10,
        # e.COIN_FOUND: 20,
        # e.CRATE_DESTROYED: 3,
        e.INVALID_ACTION: -2,
        # e.SURVIVED_ROUND: 100,
        e.MOVED_LEFT: 2,
        e.MOVED_DOWN: 2,
        e.MOVED_RIGHT: 2,
        e.MOVED_UP: 2,
        e.WAITED: -2,
        # CLOSER_TO_COIN: 5,
        # FURTHER_FROM_COIN: -1
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum


def closer_to_coin(old_game_state, new_game_state) -> bool:
    old_coins = np.array(old_game_state["coins"]) - 1
    new_coins = np.array(new_game_state["coins"]) - 1

    # Check if coin was present and is still present
    if old_coins != [] and new_coins != []:

        old_pos = np.array(old_game_state["self"][-1]) - 1
        new_pos = np.array(new_game_state["self"][-1]) - 1

        old_distance = np.linalg.norm(old_coins - old_pos, axis=1)
        new_distance = np.linalg.norm(new_coins - new_pos, axis=1)

        index_old_coin = np.argmin(np.linalg.norm(old_coins - old_pos, axis=1))
        index_new_coin = np.argmin(np.linalg.norm(new_coins - old_pos, axis=1))

        # Check if it is the same coin
        if (old_coins[index_old_coin] == new_coins[index_new_coin]).all():

            if min(old_distance) > min(new_distance):
                return True
    return False
