from copy import deepcopy

import numpy as np

import torch.nn as nn
from mushroom_rl.core.agent import Agent
from mushroom_rl.approximators import Regressor

from replay_memory import ReplayMemory


class ActorLoss(nn.Module):
    def __init__(self, critic):
        super().__init__()

        self._critic = critic

    def forward(self, arg, state, idxs):
        action = arg

        q = self._critic.model.network(state, action, idx=idxs)

        return -q.mean()


class DDPG(Agent):
    def __init__(self, actor_approximator, critic_approximator, policy_class,
                 mdp_info, batch_size, initial_replay_size, max_replay_size,
                 tau, actor_params, critic_params, policy_params,
                 n_actions_per_head, history_length=1, n_input_per_mdp=None,
                 n_games=1, batch_size_td=1000, gamma_sampling=0.99,
                 target_update_frequency=1, dtype=np.uint8):
        self._batch_size = batch_size
        self._n_games = n_games
        if n_input_per_mdp is None:
            self._n_input_per_mdp = [mdp_info.observation_space.shape
                                     for _ in range(self._n_games)]
        else:
            self._n_input_per_mdp = n_input_per_mdp
        self._n_actions_per_head = n_actions_per_head
        self._max_actions = max(n_actions_per_head)[0]
        self._history_length = history_length
        self._target_update_frequency = target_update_frequency
        self._tau = tau

        self._replay_memory = [
            ReplayMemory(initial_replay_size,
                         max_replay_size) for _ in range(self._n_games)
        ]

        self._n_updates = 0
        self._n_target_updates = 0

        target_critic_params = deepcopy(critic_params)
        self._critic_approximator = Regressor(critic_approximator,
                                              **critic_params)
        self._target_critic_approximator = Regressor(critic_approximator,
                                                     **target_critic_params)

        if 'loss' not in actor_params:
            actor_params['loss'] = ActorLoss(self._critic_approximator)

        target_actor_params = deepcopy(actor_params)
        self._actor_approximator = Regressor(actor_approximator,
                                             n_fit_targets=2, **actor_params)
        self._target_actor_approximator = Regressor(actor_approximator,
                                                    n_fit_targets=2,
                                                    **target_actor_params)

        self._target_actor_approximator.model.set_weights(
            self._actor_approximator.model.get_weights())
        self._target_critic_approximator.model.set_weights(
            self._critic_approximator.model.get_weights())

        policy = policy_class(self._actor_approximator, **policy_params)

        super().__init__(mdp_info, policy)

        n_samples = self._batch_size * self._n_games
        self._state_idxs = np.zeros(n_samples, dtype=np.int)
        self._state = np.zeros(
            ((n_samples,
             self._history_length) + self.mdp_info.observation_space.shape),
            dtype=dtype
        ).squeeze()
        self._action = np.zeros((n_samples, self._max_actions))
        self._reward = np.zeros(n_samples)
        self._next_state_idxs = np.zeros(n_samples, dtype=np.int)
        self._next_state = np.zeros(
            ((n_samples,
             self._history_length) + self.mdp_info.observation_space.shape),
            dtype=dtype
        ).squeeze()
        self._absorbing = np.zeros(n_samples)

        self._batch_size_td = batch_size_td
        assert self._batch_size_td < max_replay_size
        self._td_errors = [[] for i in range(self._n_games)]
        self._lp = np.zeros(self._n_games)
        self._lp_probabilities = np.ones(self._n_games) / self._n_games
        self._mdp_min_max = np.array([[0, -1, -1, -1, -1],
                                      [1,  0,  0,  1,  0]])
        self._gamma_sampling = gamma_sampling

    def fit(self, dataset):
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
                self._action[start:stop, :self._n_actions_per_head[i][0]] = game_action
                self._reward[start:stop] = game_reward
                self._next_state_idxs[start:stop] = np.ones(self._batch_size) * i
                self._next_state[start:stop, :self._n_input_per_mdp[i][0]] = game_next_state
                self._absorbing[start:stop] = game_absorbing

            q_next = self._next_q()
            q = self._reward + q_next

            self._critic_approximator.fit(self._state, self._action, q,
                                          idx=self._state_idxs)
            self._actor_approximator.fit(self._state, self._state,
                                         self._state_idxs,
                                         idx=self._state_idxs)

            self._n_updates += 1

            if self._n_updates % self._target_update_frequency == 0:
                self._n_target_updates += 1
                self._update_target()

    def get_shared_weights(self):
        cw = self._critic_approximator.model.network.get_shared_weights()
        aw = self._actor_approximator.model.network.get_shared_weights()

        return [cw, aw]

    def set_shared_weights(self, weights):
        self._critic_approximator.model.network.set_shared_weights(weights[0])
        self._actor_approximator.model.network.set_shared_weights(weights[1])

    def freeze_shared_weights(self):
        self._critic_approximator.model.network.freeze_shared_weights()
        self._actor_approximator.model.network.freeze_shared_weights()

    def unfreeze_shared_weights(self):
        self._critic_approximator.model.network.unfreeze_shared_weights()
        self._actor_approximator.model.network.unfreeze_shared_weights()

    def _update_target(self):
        """
        Update the target networks.

        """
        critic_weights = self._tau * self._critic_approximator.model.get_weights()
        critic_weights += (1 - self._tau) * self._target_critic_approximator.get_weights()
        self._target_critic_approximator.set_weights(critic_weights)

        actor_weights = self._tau * self._actor_approximator.model.get_weights()
        actor_weights += (1 - self._tau) * self._target_actor_approximator.get_weights()
        self._target_actor_approximator.set_weights(actor_weights)

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
        pass

    def _normalize_lp(self, lp):
        pass

    def _compute_lp(self, norm):
        pass

    def update_lp(self, norm=False, tasks=None):
        pass

    def _next_q(self, next_state=None, next_state_idxs=None, absorbing=None, target_approx=True):
        if next_state is None:
            next_state = self._next_state
            next_state_idxs = self._next_state_idxs
            absorbing = self._absorbing

        if target_approx:
            a = self._target_actor_approximator(self._next_state,
                                                idx=self._next_state_idxs)
            q = self._target_critic_approximator(self._next_state, a,
                                                idx=self._next_state_idxs).ravel()
        else:
            a = self._actor_approximator(self._next_state,
                                                idx=self._next_state_idxs)
            q = self._critic_approximator(self._next_state, a,
                                                idx=self._next_state_idxs).ravel()

        out_q = np.zeros(len(next_state_idxs))

        _, counts = np.unique(next_state_idxs, return_counts=True)
        cum_counts = np.cumsum(counts) - counts
        for i, (n_samp_in_mdp, n_samp_until_mdp) in enumerate(zip(counts, cum_counts)):
            start = n_samp_until_mdp
            stop = start + n_samp_in_mdp

            out_q[start:stop] = q[start:stop] * self.mdp_info.gamma[i]
            if np.any(self._absorbing[start:stop]):
                out_q[start:stop] *= 1 - self._absorbing[start:stop]

        return out_q
