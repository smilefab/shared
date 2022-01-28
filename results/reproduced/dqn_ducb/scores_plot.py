import numpy as np
import scipy.stats as st
from matplotlib import pyplot as plt
import os

os.chdir('C:/Users/Fabian/Documents/Master/LR - IP/Project/shared/results/reproduced/dqn_ducb')

def get_mean_and_confidence(data):
    mean = np.mean(data, axis=0)
    se = st.sem(data, axis=0)
    n = len(data)

    interval, _ = st.t.interval(0.95, n-1, scale=se)

    return mean, interval

folders = ['dqn_ducb', 'multidqn', 'dqn_prism']
games = ['Cart-Pole', 'Acrobot', 'Mountain-Car', 'Car-On-Hill', 'Inverted-Pendulum']
reg = ['noreg']
activation = ['sigmoid']
n_games = len(games)
n_settings = len(reg) * len(activation)


with np.errstate(invalid='ignore'):
    for i, g in enumerate(games):
        j = 1
        for act in activation:
            for r in reg:
                s = r + '-' + act
                plt.subplot(n_settings, n_games, i * n_settings + j)
                plt.title(g, fontsize=20)
                
                
                multi = np.load('multidqn/' + s + '/scores.npy')[:, i]
                multi_mean, multi_err = get_mean_and_confidence(multi)

                prism = np.load('dqn_prism/' + s + '/scores.npy')[:, i]
                prism_mean, prism_err = get_mean_and_confidence(prism)

                ducb = np.load('dqn_ducb/' + s + '/scores.npy')[:, i]
                ducb_mean, ducb_err = get_mean_and_confidence(ducb)
                
                plt.plot(multi_mean, linewidth=3)
                plt.fill_between(np.arange(51), multi_mean - multi_err, multi_mean + multi_err, alpha=.5, label='_nolegend_')

                plt.plot(prism_mean, linewidth=3)
                plt.fill_between(np.arange(51), prism_mean - prism_err, prism_mean + prism_err, alpha=.5, label='_nolegend_')

                plt.plot(ducb_mean, linewidth=3)
                plt.fill_between(np.arange(51), ducb_mean - ducb_err, ducb_mean + ducb_err, alpha=.5, label='_nolegend_')

                plt.xlabel('#Epochs', fontsize=20)
                plt.xticks([0, 25, 50], fontsize=20)
                plt.yticks(fontsize=20)

                if i == 0:
                    plt.ylabel('Performance', fontsize=20)
                
                plt.grid()
                
                # plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
                
                j += 1

    plt.legend(['MULTI', 'PRISM', 'D-UCB'], fontsize=20, loc='lower right')

    tasks = np.load('dqn_ducb/noreg-sigmoid/tasks.npy')
    tasks = tasks.reshape((tasks.shape[0], -1, n_games))
    occurances = np.zeros(tasks.shape)
    for n in range(tasks.shape[0]):
        for t in range(tasks.shape[1]):
            idxs, counts = np.unique(tasks[n, t], return_counts=True)
            occurances[n, t, idxs] = counts
    cum_sum = np.cumsum(occurances, axis=1)
    tasks_means, tasks_errs = get_mean_and_confidence(cum_sum)

plt.figure()
plt.title('Cumulative Sum of Sampled Tasks', fontsize=20)
epoch_factor = 1000 / n_games
epochs = (np.arange(cum_sum.shape[1]) + 1) / epoch_factor
for mean, err in zip(tasks_means.T, tasks_errs.T):
    plt.plot(epochs, mean, linewidth=3)
    plt.fill_between(epochs, mean - err, mean + err, alpha=.5, label='_nolegend_')
plt.plot(epochs, np.arange(len(epochs)), linewidth=3, color='black', label='Uniform')

plt.xlabel('#Epochs', fontsize=20)
plt.ylabel('Cumulative Sum', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.legend(games, fontsize=20, loc='lower right')

plt.show()

