from copy import deepcopy

import numpy as np

import torch.nn as nn
import torch.nn.functional as F

from mushroom_rl.core.agent import Agent
from mushroom_rl.approximators.regressor import Regressor
from mushroom_rl.approximators.parametric.torch_approximator import *

from replay_memory import PrioritizedReplayMemory, ReplayMemory


class DQN(Agent):
    """
    Deep Q-Network algorithm.
    "Human-Level Control Through Deep Reinforcement Learning".
    Mnih V. et al.. 2015.

    """
    def __init__(self, approximator, policy, mdp_info, batch_size,
                 initial_replay_size, max_replay_size, n_actions_per_head,
                 history_length=4, n_input_per_mdp=None, replay_memory=None,
                 target_update_frequency=2500, fit_params=None,
                 approximator_params=None, n_games=1, clip_reward=True,
                 batch_size_td=1000, gamma_sampling=0.99, dtype=np.uint8):
        self._fit_params = dict() if fit_params is None else fit_params

        self._batch_size = batch_size
        self._n_games = n_games
        self._clip_reward = clip_reward
        if n_input_per_mdp is None:
            self._n_input_per_mdp = [mdp_info.observation_space.shape
                                     for _ in range(self._n_games)]
        else:
            self._n_input_per_mdp = n_input_per_mdp
        self._n_action_per_head = n_actions_per_head
        self._history_length = history_length
        self._max_actions = max(n_actions_per_head)[0]
        self._target_update_frequency = target_update_frequency

        if replay_memory is not None:
            self._replay_memory = replay_memory
            if isinstance(replay_memory[0], PrioritizedReplayMemory):
                self._fit = self._fit_prioritized
            else:
                self._fit = self._fit_standard
        else:
            self._replay_memory = [ReplayMemory(
                initial_replay_size, max_replay_size) for _ in range(self._n_games)
            ]
            self._fit = self._fit_standard

        self._n_updates = 0

        apprx_params_train = deepcopy(approximator_params)
        apprx_params_target = deepcopy(approximator_params)
        self.approximator = Regressor(approximator, **apprx_params_train)
        self.target_approximator = Regressor(approximator,
                                             **apprx_params_target)
        policy.set_q(self.approximator)

        self.target_approximator.model.set_weights(
            self.approximator.model.get_weights())

        super().__init__(mdp_info, policy)

        n_samples = self._batch_size * self._n_games
        self._state_idxs = np.zeros(n_samples, dtype=np.int)
        self._state = np.zeros(
            ((n_samples,
             self._history_length) + self.mdp_info.observation_space.shape),
            dtype=dtype
        ).squeeze()
        self._action = np.zeros((n_samples, 1), dtype=np.int)
        self._reward = np.zeros(n_samples)
        self._next_state_idxs = np.zeros(n_samples, dtype=np.int)
        self._next_state = np.zeros(
            ((n_samples,
             self._history_length) + self.mdp_info.observation_space.shape),
            dtype=dtype
        ).squeeze()
        self._absorbing = np.zeros(n_samples)
        self._idxs = np.zeros(n_samples, dtype=np.int)
        self._is_weight = np.zeros(n_samples)

        self._batch_size_td = batch_size_td
        assert self._batch_size_td < max_replay_size
        self._td_errors = [[] for i in range(self._n_games)]
        self._lp = np.zeros(self._n_games)
        self._lp_probabilities = np.ones(self._n_games) / self._n_games
        self._mdp_min_max = np.array([[0, -1, -1, -1, -1],
                                      [1,  0,  0,  1,  0]])
        self._gamma_sampling = gamma_sampling


    def fit(self, dataset):
        self._fit(dataset)

        self._n_updates += 1
        if self._n_updates % self._target_update_frequency == 0:
            self._update_target()

    def _fit_standard(self, dataset):
        s = np.array([d[0][0] for d in dataset]).ravel()
        games = np.unique(s)
        for g in games:
            idxs = np.argwhere(s == g).ravel()
            d = list()
            for idx in idxs:
                d.append(dataset[idx])

            self._replay_memory[g].add(d)

        fit_condition = np.all([rm.initialized for rm in self._replay_memory])

        if fit_condition:
            for i in range(len(self._replay_memory)):
                game_state, game_action, game_reward, game_next_state,\
                    game_absorbing, _ = self._replay_memory[i].get(
                        self._batch_size)

                start = self._batch_size * i
                stop = start + self._batch_size

                self._state_idxs[start:stop] = np.ones(self._batch_size) * i
                self._state[start:stop, :self._n_input_per_mdp[i][0]] = game_state
                self._action[start:stop] = game_action
                self._reward[start:stop] = game_reward
                self._next_state_idxs[start:stop] = np.ones(self._batch_size) * i
                self._next_state[start:stop, :self._n_input_per_mdp[i][0]] = game_next_state
                self._absorbing[start:stop] = game_absorbing

            if self._clip_reward:
                reward = np.clip(self._reward, -1, 1)
            else:
                reward = self._reward

            q_next = self._next_q()
            q = reward + q_next

            self.approximator.fit(self._state, self._action, q,
                                  idx=self._state_idxs, **self._fit_params)

    def _fit_prioritized(self, dataset):
        s = np.array([d[0][0] for d in dataset]).ravel()
        games = np.unique(s)
        for g in games:
            idxs = np.argwhere(s == g).ravel()
            d = list()
            for idx in idxs:
                d.append(dataset[idx])

            self._replay_memory[g].add(
                d, np.ones(len(d)) * self._replay_memory[g].max_priority
            )

        fit_condition = np.all([rm.initialized for rm in self._replay_memory])

        if fit_condition:
            for i in range(len(self._replay_memory)):
                game_state, game_action, game_reward, game_next_state,\
                    game_absorbing, _, game_idxs, game_is_weight =\
                    self._replay_memory[i].get(self._batch_size)

                start = self._batch_size * i
                stop = start + self._batch_size

                self._state_idxs[start:stop] = np.ones(self._batch_size) * i
                self._state[start:stop, :self._n_input_per_mdp[i][0]] = game_state
                self._action[start:stop] = game_action
                self._reward[start:stop] = game_reward
                self._next_state_idxs[start:stop] = np.ones(self._batch_size) * i
                self._next_state[start:stop, :self._n_input_per_mdp[i][0]] = game_next_state
                self._absorbing[start:stop] = game_absorbing
                self._idxs[start:stop] = game_idxs
                self._is_weight[start:stop] = game_is_weight

            if self._clip_reward:
                reward = np.clip(self._reward, -1, 1)
            else:
                reward = self._reward

            q_next = self._next_q()
            q = reward + q_next
            q_current = self.approximator.predict(self._state, self._action,
                                                  idx=self._state_idxs)
            td_error = q - q_current

            for er in self._replay_memory:
                er.update(td_error, self._idxs)

            self.approximator.fit(self._state, self._action, q,
                                  weights=self._is_weight,
                                  idx=self._state_idxs,
                                  **self._fit_params)

    def get_shared_weights(self):
        return self.approximator.model.network.get_shared_weights()

    def set_shared_weights(self, weights):
        self.approximator.model.network.set_shared_weights(weights)

    def freeze_shared_weights(self):
        self.approximator.model.network.freeze_shared_weights()

    def unfreeze_shared_weights(self):
        self.approximator.model.network.unfreeze_shared_weights()

    def _update_target(self):
        """
        Update the target network.

        """
        self.target_approximator.model.set_weights(
            self.approximator.model.get_weights())

    def _get_samples_from_replay_memory(self, tasks=None):
        tasks = np.arange(self._n_games) if tasks is None else tasks
        n_samples = np.zeros(self._n_games, np.int64)

        # Getting amount of samples from each task in the replay memory
        for i in tasks:
            if self._replay_memory[i]._full:
                n_samples[i] = self._batch_size_td
            else:
                n_samples[i] = min(int(self._replay_memory[i]._idx), self._batch_size_td)

        # Initializing variables
        state_dim = self._state.shape[1]
        state_idxs = np.zeros(sum(n_samples), np.int64)
        state = np.zeros((sum(n_samples), state_dim))
        action = np.zeros((sum(n_samples), 1), np.int64)
        reward = np.zeros(sum(n_samples))
        next_state_idxs = np.zeros(sum(n_samples), np.int64)
        next_state = np.zeros((sum(n_samples), state_dim))
        absorbing = np.zeros(sum(n_samples))
        start = 0
        stop = 0

        # Getting samples out of the replay memory
        for i in tasks:
            game_state, game_action, game_reward, game_next_state, game_absorbing, _ = self._replay_memory[i].get(
                n_samples[i])

            stop = stop + n_samples[i]

            state_idxs[start:stop] = np.ones(n_samples[i]) * i
            state[start:stop, :self._n_input_per_mdp[i][0]] = game_state
            action[start:stop] = game_action
            reward[start:stop] = game_reward
            next_state_idxs[start:stop] = np.ones(n_samples[i]) * i
            next_state[start:stop, :self._n_input_per_mdp[i][0]] = game_next_state
            absorbing[start:stop] = game_absorbing

            start = stop
        
        return state, state_idxs, action, reward, next_state, next_state_idxs, absorbing

    def _update_td_errors(self, tasks):
        state, state_idxs, action, reward, next_state, next_state_idxs, absorbing = self._get_samples_from_replay_memory(tasks)
        q_next = self._next_q(next_state, next_state_idxs, absorbing, target_approx=False)
        q = reward + q_next
        q_current = self.approximator.predict(state, action, idx=state_idxs)
        
        td_errors = (q - q_current)
        for idx in tasks:
            self._td_errors[idx] = td_errors[state_idxs==idx]
        
    def _normalize_lp(self, lp): # TODO: make cleaner using numpy magic
        normalized = []
        for i in range(self._n_games):
            R_max = self._mdp_min_max[1, i]
            R_min = self._mdp_min_max[0, i]
            normalized.append(lp[i] / ((R_max - R_min)/(1-self._gamma_sampling)))
        return np.array(normalized).copy()

    def _compute_lp(self, norm):
        td_errors = np.array([np.mean(np.array(a)) for a in self._td_errors])   # TODO: optimize: only update new values in td_errors
        abs_td_errors = np.abs(td_errors)
        if norm:
            self._lp = self._normalize_lp(abs_td_errors) if norm else abs_td_errors
        self._lp_probabilities = abs_td_errors / np.sum(abs_td_errors)
        assert not True in np.isnan(self._lp_probabilities), \
            f'NAN encountered \n lp_probas: {self._lp_probabilities}' \
            f'isnan(): {np.isnan(self._lp_probabilities)}'

    def update_lp(self, norm=False, tasks=None):
        if tasks is None:
            tasks = np.arange(self._n_games)
        elif type(tasks) != list:
            tasks = [tasks]
        if not True in [self._replay_memory[i].size == 0 for i in tasks]:
            self._update_td_errors(tasks)
            self._compute_lp(norm)


    def _next_q(self, next_state=None, next_state_idxs=None, absorbing=None, target_approx=True):
        if next_state is None:
            next_state = self._next_state
            next_state_idxs = self._next_state_idxs
            absorbing = self._absorbing
        
        if target_approx:
            q = self.target_approximator.predict(next_state,
                                             idx=next_state_idxs)
        else:
            q = self.approximator.predict(next_state,
                                            idx=next_state_idxs)

        out_q = np.zeros(len(next_state_idxs))

        _, counts = np.unique(next_state_idxs, return_counts=True)
        cum_counts = np.cumsum(counts) - counts
        for i, (n_samp_in_mdp, n_samp_until_mdp) in enumerate(zip(counts, cum_counts)):
            start = n_samp_until_mdp
            stop = start + n_samp_in_mdp
            if np.any(absorbing[start:stop]):
                q[start:stop] *= 1 - absorbing[start:stop].reshape(-1, 1)

            n_actions = self._n_action_per_head[i][0]
            out_q[start:stop] = np.max(q[start:stop, :n_actions], axis=1)
            out_q[start:stop] *= self.mdp_info.gamma[i]
        
        return out_q

        
class DoubleDQN(DQN):
    """
    Double DQN algorithm.
    "Deep Reinforcement Learning with Double Q-Learning".
    Hasselt H. V. et al.. 2016.

    """
    def _next_q(self):
        q = self.approximator.predict(self._next_state,
                                      idx=self._next_state_idxs)
        out_q = np.zeros(self._batch_size * self._n_games)

        for i in range(self._n_games):
            start = self._batch_size * i
            stop = start + self._batch_size
            n_actions = self._n_action_per_head[i][0]
            max_a = np.argmax(q[start:stop, :n_actions], axis=1)

            double_q = self.target_approximator.predict(
                self._next_state[start:stop], max_a,
                idx=self._next_state_idxs[start:stop]
            )
            if np.any(self._absorbing[start:stop]):
                double_q *= 1 - self._absorbing[start:stop].reshape(-1, 1)

            out_q[start:stop] = double_q * self.mdp_info.gamma[i]

        return out_q
