from numpy.core.fromnumeric import argmax
from tqdm import tqdm
import numpy as np


class Core(object):
    def __init__(self, agent, mdp, sampling, epsilon_samp,
                 lp_update_frequency=1, callbacks=None):
        self.agent = agent
        self.mdp = mdp
        self._n_mdp = len(self.mdp)
        self.callbacks = callbacks if callbacks is not None else list()
        self._init = True

        self._state = [None for _ in range(self._n_mdp)]

        self._total_learning_steps = 0
        self._epoch_steps = 0
        self._steps_since_last_fit = 0
        self._episode_steps = [None for _ in range(self._n_mdp)]
        self._n_steps_per_fit = None

        self._sampling = sampling
        self._norm = True if sampling in ['d-ucb', 't-d-ucb'] else False
        self._epsilon_samp = epsilon_samp

        self.update_lp_condition = lambda: False
        self._lp_update_frequency = lp_update_frequency
        self._N_t = np.zeros(self._n_mdp)
        self._X_t_sum = np.zeros(self._n_mdp)
        self._tasks_list = []
        self._gamma_sampling = agent._gamma_sampling

    def learn(self, n_steps=None, n_steps_per_fit=None, render=False,
              quiet=False):
        self._n_steps_per_fit = n_steps_per_fit

        fit_condition = \
            lambda: self._steps_since_last_fit >= self._n_steps_per_fit
        
        self.update_lp_condition = lambda: self._total_learning_steps % self._lp_update_frequency == 0 \
            and self._sampling != 'uniform' and not self._init
        self._run(n_steps, fit_condition, render, quiet, eval=False)
        if self._init:
            self.agent.update_lp(self._norm)
            self._update_ucb_params(np.arange(self._n_mdp))
            self._init = False


    def evaluate(self, n_steps=None, render=False,
                 quiet=False):
        fit_condition = lambda: False

        self.update_lp_condition = lambda: False

        return self._run(n_steps, fit_condition, render, quiet, eval=True)

    def _run(self, n_steps, fit_condition, render, quiet, eval):
        move_condition = lambda: self._epoch_steps < n_steps

        steps_progress_bar = tqdm(total=n_steps,
                                  dynamic_ncols=True, disable=quiet,
                                  leave=False)
        steps_progress_bar = None

        return self._run_impl(move_condition, fit_condition, steps_progress_bar,
                              render, eval)

    def _update_ucb_params(self, tasks):
        n_new = np.zeros(self._n_mdp)
        x_new = np.zeros(self._n_mdp)
        for task in tasks:
            n_new[task] = 1
            x_new[task] = self.agent._lp[task]
        self._N_t = self._N_t * self._gamma_sampling + n_new
        self._X_t_sum = self._X_t_sum * self._gamma_sampling + x_new

    def _sample_tasks(self, eval):
        if eval or self._init:
            return list(np.arange(self._n_mdp))
        if self._sampling == 'uniform':
            self._tasks_list.append(list(np.arange(self._n_mdp)))
        elif self._sampling == 'prism':
            eps = self._epsilon_samp()
            probas = eps / self._n_mdp \
                + (1 - eps) * self.agent._lp_probabilities
            sampled_tasks = np.random.choice(self._n_mdp, self._n_mdp, p=probas)
            self._tasks_list.append(list(sampled_tasks))
        elif self._sampling == 'd-ucb':
            B = 0.5
            Xi = 0.002
            X_t = self._X_t_sum / self._N_t
            n_t = np.sum(self._N_t)
            c_t = 2 * B * np.sqrt(Xi * np.log(n_t) / self._N_t)
            self._tasks_list.append([np.argmax(X_t + c_t)])
        elif self._sampling == 't-d-ucb':
            X_t = self._X_t_sum / self._N_t
            n_t = np.sum(self._N_t)
            Xi = np.max(np.concatenate((np.multiply(X_t, (1 - X_t)) , np.array([0.002])), axis=0))
            c_t = np.sqrt(Xi * np.log(n_t) / self._N_t)
            self._tasks_list.append([np.argmax(X_t + c_t)])
        assert self._tasks_list[-1] is not None, 'problem with task sampling'
        return self._tasks_list[-1]
    

    def _run_impl(self, move_condition, fit_condition, steps_progress_bar,
                  render, eval):
        self._epoch_steps = 0
        self._steps_since_last_fit = 0

        dataset = list()
        last = [True] * self._n_mdp
        while move_condition():
            tasks = self._sample_tasks(eval)
            for i in tasks:
                if last[i]:
                    self.reset(i)

                sample = self._step(i, render)
                dataset.append(sample)

                last[i] = sample[-1]

            if not eval:
                self._total_learning_steps += 1
            self._epoch_steps += 1
            self._steps_since_last_fit += 1
            # steps_progress_bar.update(1)

            
            if self.update_lp_condition():
                self.agent.update_lp(self._norm, tasks)
                if self._sampling in ['d-ucb', 't-d-ucb']:
                    self._update_ucb_params(tasks)

            if fit_condition():
                self.agent.fit(dataset)
                self._steps_since_last_fit = 0

                for c in self.callbacks:
                    callback_pars = dict(dataset=dataset)
                    c(**callback_pars)

                dataset = list()

        self.agent.stop()
        for i in range(self._n_mdp):
            self.mdp[i].stop()

        return dataset

    def _step(self, i, render):
        action = self.agent.draw_action([i, self._state[i]])
        next_state, reward, absorbing, _ = self.mdp[i].step(action)

        self._episode_steps[i] += 1

        if render:
            self.mdp[i].render()

        last = not(
            self._episode_steps[i] < self.mdp[i].info.horizon and not absorbing)

        state = self._state[i]
        self._state[i] = next_state.copy()

        return [i, state], action, reward, [i, next_state], absorbing, last

    def reset(self, i):
        self._state[i] = self.mdp[i].reset().copy()
        self.agent.episode_start()
        self.agent.next_action = None
        self._episode_steps[i] = 0
