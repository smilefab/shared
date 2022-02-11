import argparse
import datetime
import pathlib
import sys

from joblib import delayed, Parallel
import numpy as np
import torch.optim as optim

import pickle

sys.path.append('..')
sys.path.append('.')

from mushroom_rl.approximators.parametric.torch_approximator import TorchApproximator
from mushroom_rl.core.environment import MDPInfo
from mushroom_rl.environments import *
from mushroom_rl.utils.dataset import compute_J
from mushroom_rl.utils.parameters import LinearParameter, Parameter

from core import Core
from ddpg import DDPG
from policy import OrnsteinUhlenbeckPolicy
from networks import ActorNetwork, CriticNetwork
from losses import LossFunction
from replay_memory import PrioritizedReplayMemory

"""
This script runs Atari experiments with DQN as presented in:
"Human-Level Control Through Deep Reinforcement Learning". Mnih V. et al.. 2015.

"""


def print_epoch(epoch):
    print('################################################################')
    print('Epoch: ', epoch)
    print('----------------------------------------------------------------')


def get_stats(dataset, gamma, idx, games):
    J = np.mean(compute_J(dataset, gamma[idx]))
    print(games[idx] + ': J: %f' % J)

    return J


def experiment(args, idx):
    np.random.seed()

    args.games = [''.join(g) for g in args.games]

    # MDP
    mdp = list()
    gamma_eval = list()
    for i, g in enumerate(args.games):
        if g == 'pendulum':
            mdp.append(CartPole(horizon=args.horizon[i], gamma=args.gamma[i]))
        elif g == 'caronhill':
            mdp.append(CarOnHill(horizon=args.horizon[i], gamma=args.gamma[i]))
        else:
            mdp.append(Gym(g, args.horizon[i], args.gamma[i]))

        gamma_eval.append(args.gamma[i])

    n_input_per_mdp = [m.info.observation_space.shape for m in mdp]
    n_actions_per_head = [(m.info.action_space.n,) for m in mdp]

    max_obs_dim = 0
    max_act_n = 0
    for i in range(len(args.games)):
        n = mdp[i].info.observation_space.shape[0]
        m = mdp[i].info.action_space.n
        if n > max_obs_dim:
            max_obs_dim = n
            max_obs_idx = i
        if m > max_act_n:
            max_act_n = m
            max_act_idx = i
    gammas = [m.info.gamma for m in mdp]
    horizons = [m.info.horizon for m in mdp]
    mdp_info = MDPInfo(mdp[max_obs_idx].info.observation_space,
                       mdp[max_act_idx].info.action_space, gammas, horizons)
    max_action_value = list()
    for m in mdp:
        assert len(np.unique(m.info.action_space.low)) == 1
        assert len(np.unique(m.info.action_space.high)) == 1
        assert abs(m.info.action_space.low[0]) == m.info.action_space.high[0]

        max_action_value.append(m.info.action_space.high[0])

    scores = list()
    for _ in range(len(args.games)):
        scores.append(list())

    optimizer_actor = dict()
    optimizer_critic = dict()
    if args.optimizer == 'adam':    
        optimizer_actor['class'] = optim.Adam
        optimizer_actor['params'] = dict(lr=args.learning_rate_actor)
        optimizer_critic['class'] = optim.Adam
        optimizer_critic['params'] = dict(lr=args.learning_rate_critic)
    elif args.optimizer == 'adadelta':
        optimizer_actor['class'] = optim.Adadelta
        optimizer_actor['params'] = dict(lr=args.learning_rate_actor)
        optimizer_critic['class'] = optim.Adadelta
        optimizer_critic['params'] = dict(lr=args.learning_rate_critic)
    elif args.optimizer == 'rmsprop':
        optimizer_actor['class'] = optim.RMSprop
        optimizer_actor['params'] = dict(lr=args.learning_rate_actor,
                                         alpha=args.decay)
        optimizer_critic['class'] = optim.RMSprop
        optimizer_critic['params'] = dict(lr=args.learning_rate_critic,
                                          alpha=args.decay)
    elif args.optimizer == 'rmspropcentered':
        optimizer_actor['class'] = optim.RMSprop
        optimizer_actor['params'] = dict(lr=args.learning_rate_actor,
                                         alpha=args.decay,
                                         eps=args.epsilon,
                                         centered=True)
        optimizer_critic['class'] = optim.RMSprop
        optimizer_critic['params'] = dict(lr=args.learning_rate_critic,
                                          alpha=args.decay,
                                          eps=args.epsilon,
                                          centered=True)
    else:
        raise ValueError

    # DQN learning run

    # Settings
    if args.debug:
        initial_replay_size = args.batch_size
        max_replay_size = 500
        train_frequency = 5
        target_update_frequency = 10
        test_samples = 20
        evaluation_frequency = 50
        max_steps = 1000
        lp_update_frequency = 1
    else:
        initial_replay_size = args.initial_replay_size
        max_replay_size = args.max_replay_size
        train_frequency = args.train_frequency
        target_update_frequency = args.target_update_frequency
        test_samples = args.test_samples
        evaluation_frequency = args.evaluation_frequency
        max_steps = args.max_steps
        lp_update_frequency = args.lp_update_frequency

    # Policy
    policy_class = OrnsteinUhlenbeckPolicy
    policy_params = dict(sigma=np.ones(1) * .2, theta=.15, dt=1e-2,
                         n_actions_per_head=n_actions_per_head,
                         max_action_value=max_action_value)

    # Approximator
    n_games = len(args.games)
    loss = LossFunction(n_games, args.batch_size,
                        args.evaluation_frequency)

    actor_approximator = TorchApproximator
    actor_input_shape = [m.info.observation_space.shape for m in mdp]
    actor_approximator_params = dict(
        network=ActorNetwork,
        input_shape=actor_input_shape,
        output_shape=(max(n_actions_per_head)[0],),
        n_actions_per_head=n_actions_per_head,
        n_hidden_1=args.hidden_neurons[0],
        n_hidden_2=args.hidden_neurons[1],
        optimizer=optimizer_actor,
        use_cuda=args.use_cuda,
        features=args.features
    )

    critic_approximator = TorchApproximator
    critic_input_shape = [m.info.observation_space.shape for m in mdp]
    critic_approximator_params = dict(
        network=CriticNetwork,
        input_shape=critic_input_shape,
        output_shape=(1,),
        n_actions_per_head=n_actions_per_head,
        n_hidden_1=args.hidden_neurons[0],
        n_hidden_2=args.hidden_neurons[1],
        optimizer=optimizer_actor,
        loss=loss,
        use_cuda=args.use_cuda,
        features=args.features
    )

    # Agent
    algorithm_params = dict(
        batch_size=args.batch_size,
        n_games=len(args.games),
        initial_replay_size=initial_replay_size,
        max_replay_size=max_replay_size,
        tau=args.tau,
        actor_params=actor_approximator_params,
        critic_params=critic_approximator_params,
        policy_params=policy_params,
        target_update_frequency=target_update_frequency // train_frequency,
        n_input_per_mdp=n_input_per_mdp,
        n_actions_per_head=n_actions_per_head,
        clip_reward=False,
        history_length=args.history_length,
        dtype=np.float32,
        batch_size_td=args.batch_size_td,
        gamma_sampling=args.gamma_sampling,
    )

    agent = DDPG(actor_approximator, critic_approximator, policy_class,
                 mdp_info, **algorithm_params)

    # Algorithm
    core = Core(agent, mdp, sampling=args.sampling, lp_update_frequency=lp_update_frequency)

    # RUN

    # Fill replay memory with random dataset
    print_epoch(0)
    core.learn(n_steps=initial_replay_size,
               n_steps_per_fit=initial_replay_size, quiet=args.quiet)

    if args.transfer:
        weights = pickle.load(open(args.transfer, 'rb'))
        agent.set_shared_weights(weights)

    if args.load:
        weights = np.load(args.load)
        agent.approximator.set_weights(weights)

    # Evaluate initial policy
    dataset = core.evaluate(n_steps=test_samples, render=args.render,
                            quiet=args.quiet)
    for i in range(len(mdp)):
        d = dataset[i::len(mdp)]
        scores[i].append(get_stats(d, gamma_eval, i, args.games))

    if args.unfreeze_epoch > 0:
        agent.freeze_shared_weights()

    best_score_sum = -np.inf
    best_weights = None

    np.save(folder_name + 'scores-exp-%d.npy' % idx, scores)
    np.save(folder_name + 'loss-exp-%d.npy' % idx,
            agent.approximator.model._loss.get_losses())

    for n_epoch in range(1, max_steps // evaluation_frequency + 1):
        if n_epoch >= args.unfreeze_epoch > 0:
            agent.unfreeze_shared_weights()

        print_epoch(n_epoch)
        print('- Learning:')
        # learning step
        core.learn(n_steps=evaluation_frequency,
                   n_steps_per_fit=train_frequency,
                   quiet=args.quiet)

        print('- Evaluation:')
        # evaluation step
        dataset = core.evaluate(n_steps=test_samples,
                                render=args.render, quiet=args.quiet)

        current_score_sum = 0
        for i in range(len(mdp)):
            d = dataset[i::len(mdp)]
            current_score = get_stats(d, gamma_eval, i, args.games)
            scores[i].append(current_score)
            current_score_sum += current_score

        # Save shared weights if best score
        if args.save_shared and current_score_sum >= best_score_sum:
            best_score_sum = current_score_sum
            best_weights = agent.get_shared_weights()

        if args.save:
            np.save(folder_name + 'weights-exp-%d-%d.npy' % (idx, n_epoch),
                    agent.approximator.get_weights())
        
        np.save(folder_name + 'scores-exp-%d.npy' % idx, scores)
        np.save(folder_name + 'loss-exp-%d.npy' % idx,
                agent.approximator.model._loss.get_losses())

    if args.save_shared:
        pickle.dump(best_weights, open(args.save_shared, 'wb'))

    return scores, agent.approximator.model._loss.get_losses(), np.array(core._tasks_list)


if __name__ == '__main__':
    # Argument parser
    parser = argparse.ArgumentParser()

    arg_game = parser.add_argument_group('Game')
    arg_game.add_argument("--games",
                          type=list,
                          nargs='+',
                          default=['Acrobot-v1'],
                          help='Gym ID of the problem.')
    arg_game.add_argument("--horizon", type=int, nargs='+')
    arg_game.add_argument("--gamma", type=float, nargs='+')
    arg_game.add_argument("--n-exp", type=int)

    arg_mem = parser.add_argument_group('Replay Memory')
    arg_mem.add_argument("--initial-replay-size", type=int, default=100,
                         help='Initial size of the replay memory.')
    arg_mem.add_argument("--max-replay-size", type=int, default=5000,
                         help='Max size of the replay memory.')
    arg_mem.add_argument("--prioritized", action='store_true',
                         help='Whether to use prioritized memory or not.')

    arg_net = parser.add_argument_group('Deep Q-Network')
    arg_net.add_argument("--optimizer",
                         choices=['adadelta',
                                  'adam',
                                  'rmsprop',
                                  'rmspropcentered'],
                         default='adam',
                         help='Name of the optimizer to use to learn.')
    arg_net.add_argument("--learning-rate-actor", type=float, default=1e-4,
                         help='Learning rate value of the optimizer. Only used'
                              'in rmspropcentered')
    arg_net.add_argument("--learning-rate-critic", type=float, default=1e-3,
                         help='Learning rate value of the optimizer. Only used'
                              'in rmspropcentered')
    arg_net.add_argument("--decay", type=float, default=.95,
                         help='Discount factor for the history coming from the'
                              'gradient momentum in rmspropcentered')
    arg_net.add_argument("--epsilon", type=float, default=1e-8,
                         help='Epsilon term used in rmspropcentered')

    arg_alg = parser.add_argument_group('Algorithm')
    arg_alg.add_argument("--algorithm", choices=['dqn', 'ddqn'],
                         default='dqn',
                         help='Name of the algorithm. dqn is for standard'
                              'DQN, ddqn is for Double DQN and adqn is for'
                              'Averaged DQN.')
    arg_alg.add_argument("--features", choices=['relu', 'sigmoid'])
    arg_alg.add_argument("--batch-size", type=int, default=100,
                         help='Batch size for each fit of the network.')
    arg_alg.add_argument("--history-length", type=int, default=1,
                         help='Number of frames composing a state.')
    arg_alg.add_argument("--target-update-frequency", type=int, default=100,
                         help='Number of collected samples before each update'
                              'of the target network.')
    arg_alg.add_argument("--evaluation-frequency", type=int, default=1000,
                         help='Number of learning step before each evaluation.'
                              'This number represents an epoch.')
    arg_alg.add_argument("--train-frequency", type=int, default=1,
                         help='Number of learning steps before each fit of the'
                              'neural network.')
    arg_alg.add_argument("--lp-update-frequency", type=int, default=100,
                         help='Number of collected samples before each update'
                              'of the learning progress (TD-error).')
    arg_alg.add_argument("--max-steps", type=int, default=50000,
                         help='Total number of learning steps.')
    arg_alg.add_argument("--final-exploration-frame", type=int, default=5000,
                         help='Number of steps until the exploration rate stops'
                              'decreasing.')
    arg_alg.add_argument("--initial-exploration-rate", type=float, default=1.,
                         help='Initial value of the exploration rate.')
    arg_alg.add_argument("--final-exploration-rate", type=float, default=.01,
                         help='Final value of the exploration rate. When it'
                              'reaches this values, it stays constant.')
    arg_alg.add_argument("--test-exploration-rate", type=float, default=0.,
                         help='Exploration rate used during evaluation.')
    arg_alg.add_argument("--test-samples", type=int, default=2000,
                         help='Number of steps for each evaluation.')
    arg_alg.add_argument("--max-no-op-actions", type=int, default=0,
                         help='Maximum number of no-op action performed at the'
                              'beginning of the episodes. The minimum number is'
                              'history_length.')
    arg_alg.add_argument("--transfer", type=str, default='',
                         help='Path to  the file of the weights of the common '
                              'layers to be loaded')
    arg_alg.add_argument("--save-shared", type=str, default='',
                         help='filename where to save the shared weights')
    arg_alg.add_argument("--unfreeze-epoch", type=int, default=0,
                         help="Number of epoch where to unfreeze shared weights.")
    arg_alg.add_argument("--sampling", default='uniform',
                         choices=['uniform', 'prism', 'd-ucb', 't-d-ucb'],
                         help='Sampling of tasks for learning process, choose '
                         'between \"uniform\", \"prism\", \"d-ucb\" and \"t-d-ucb\".')
    arg_alg.add_argument('--batch-size-td', type=int, default=1000,
                         help='Number of samples per task used to update the td '
                         'error. Not used for uniform sampling')
    arg_alg.add_argument("--gamma-sampling", type=float, default=0.99,
                         help="Gamma used for sampling new tasks. Only used "
                              "for D-UCB sampling.")                           

    arg_utils = parser.add_argument_group('Utils')
    arg_utils.add_argument('--use-cuda', action='store_true',
                           help='Flag specifying whether to use the GPU.')
    arg_utils.add_argument('--load', type=str,
                           help='Path of the model to be loaded.')
    arg_utils.add_argument('--save', action='store_true',
                           help='Flag specifying whether to save the model.')
    arg_utils.add_argument('--render', action='store_true',
                           help='Flag specifying whether to render the game.')
    arg_utils.add_argument('--quiet', action='store_true',
                           help='Flag specifying whether to hide the progress'
                                'bar.')
    arg_utils.add_argument('--debug', action='store_true',
                           help='Flag specifying whether the script has to be'
                                'run in debug mode.')
    arg_utils.add_argument('--postfix', type=str, default='',
                           help='Flag used to add a postfix to the folder name')

    # if no arguments given, take default args for prism
    if not len(sys.argv) > 1:
        args_str =  '--features sigmoid ' \
                    '--n-exp 6 ' \
                    '--game CartPole-v1 Acrobot-v1 MountainCar-v0 caronhill pendulum ' \
                    '--gamma .99 .99 .99 .95 .95 ' \
                    '--horizon 500 1000 1000 100 3000 ' \
                    '--sampling uniform' 
        # args_str =  '--features sigmoid ' \
        #             '--n-exp 6 ' \
        #             '--game CartPole-v1 Acrobot-v1 MountainCar-v0 caronhill pendulum ' \
        #             '--gamma .99 .99 .99 .95 .95 ' \
        #             '--horizon 500 1000 1000 100 3000 ' \
        #             '--sampling d-ucb ' \
        #             '--gamma-sampling 0.99 ' \
        #             '--lp-update-frequency 1 ' \
        #             '--batch-size-td 32 ' \
        #             '--evaluation-frequency 5000 ' \
        #             '--max-steps 250000'
        args = parser.parse_args(args_str.split())
    else:
        args = parser.parse_args()

    folder_name = './logs/gym_' + datetime.datetime.now().strftime(
        '%Y-%m-%d_%H-%M-%S') + args.postfix + '/'
    pathlib.Path(folder_name).mkdir(parents=True)
    with open(folder_name + 'args.pkl', 'wb') as f:
        pickle.dump(args, f)

    out = Parallel(n_jobs=-1)(delayed(experiment)(args, i)
                              for i in range(args.n_exp))

    scores = np.array([o[0] for o in out])
    loss = np.array([o[1] for o in out])
    tasks = np.array([o[2] for o in out])

    np.save(folder_name + 'scores.npy', scores)
    np.save(folder_name + 'loss.npy', loss)
    np.save(folder_name + 'tasks.npy', tasks)
