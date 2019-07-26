import sys
import argparse
import numpy as np
import tensorflow as tf
import time
import pickle
import os, gym, roboschool
import numpy as np

import maddpg.common.tf_util as U
from maddpg.trainer.maddpg import MADDPGAgentTrainer
import tensorflow.contrib.layers as layers
from maddpg.trainer.init_wb import init_init

import os, gym, roboschool
import numpy as np
from colorama import Fore, Back, Style
from experiments.policy import QFPolicy

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# import tensorflow as tf
# config = tf.ConfigProto(
#     inter_op_parallelism_threads=1,
#     intra_op_parallelism_threads=1,
#     device_count = { "GPU": 0 } )
# sess = tf.InteractiveSession(config=config)

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=200, help="maximum episode length") #25
    parser.add_argument("--num-episodes", type=int, default=600000, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="ddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="ddpg", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--qlr", type=float, default=5e-3, help="learning rate for Adam optimizer")
    parser.add_argument("--plr", type=float, default=1e-3, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=512, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default='pg', help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="/home/lsq/multiagent/pong-maddpg/experiments/", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=100, help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="", help="directory in which training state and model are loaded")
    # Evaluation
    # parser.add_argument("--restore", action="store_true", default=True)
    # parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="benchmark_files", help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="dd", help="directory where plot data is saved")
    return parser.parse_args()

def mlp_modelp(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None, acti=False):
    # This model takes as input an observation and returns values of all actions
    # wa,ba,wb,bb,wc,bc = init_init()
    # print(num_outputs)
    with tf.variable_scope(scope, reuse=reuse):
        # out = input
        # out = tf.layers.dense(out, units=64, activation=tf.nn.relu, kernel_initializer=wa,
        #                     bias_initializer=ba)
        # out = tf.layers.dense(out, units=32, activation=tf.nn.relu, kernel_initializer=wb,
        #                     bias_initializer=bb)
        # out = tf.layers.dense(out, units=1, activation=None, kernel_initializer=wc,
        #                     bias_initializer=bc)
        out = input
        out = tf.layers.dense(out, units=64, activation=tf.nn.relu)
        out = tf.layers.dense(out, units=32, activation=tf.nn.relu)
        out = tf.layers.dense(out, units=1, activation=None)
        return out
def mlp_modelq(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        # out = input
        # out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        # out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        # out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        out = input
        out = tf.layers.dense(out, units=64, activation=tf.nn.relu)
        out = tf.layers.dense(out, units=32, activation=tf.nn.relu)
        out = tf.layers.dense(out, units=num_outputs, activation=None)
        return out


def get_trainers(env, num_adversaries, obs_shape_n, arglist):
    trainers = []
    model = [mlp_modelq, mlp_modelp]
    # actmodel = actmlp_model
    trainer = MADDPGAgentTrainer
    # actspace = [gym.spaces.Discrete(2), gym.spaces.Discrete(2)]
    actspace = [env.action_space, env.action_space]
    # print(actspace)
    for i in range(num_adversaries):
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.adv_policy=='ddpg')))
    for i in range(num_adversaries, 2):
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, actspace, i, arglist,
            local_q_func=(arglist.good_policy=='ddpg')))
    return trainers


def train(arglist):
    pong = roboschool.gym_pong.PongSceneMultiplayer()

    participants = []
    for lane in range(2):
        env_id = "RoboschoolPong-v1"
        env = gym.make(env_id)
        env.unwrapped.scene = pong  # if you set scene before first reset(), it will be used.
        env.unwrapped.player_n = lane  # mutliplayer scenes will also use player_n
        # pi = PolicyClass(env.observation_space, env.action_space)
        participants.append(env)

    seed = 1
    for env in participants:
        np.random.seed(seed)
        env.seed(seed)

    obs_shape_n = [env.observation_space.shape, env.observation_space.shape]
    # print(obs_shape_n)
    # print(env.observation_space.shape)
    num_adversaries = min(2, arglist.num_adversaries)
    trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)
    print('Using good policy {} and adv policy {}'.format(arglist.good_policy, arglist.adv_policy))

    final_ag0_rewards = []  # sum of rewards for training curve
    final_ag1_rewards = []  # agent rewards for training curve

    for i in range(300):
        if len(final_ag0_rewards) > 0:
            # print(len(episode_rewards))
            rew_file_name = 'ag0_rewards.pkl'
            with open(rew_file_name, 'wb') as fp:
                pickle.dump(final_ag0_rewards, fp)
            agrew_file_name = 'ag1_rewards.pkl'
            with open(agrew_file_name, 'wb') as fp:
                pickle.dump(final_ag1_rewards, fp)
            print('Saving data')
        with U.single_threaded_session():
            # Create environment
            # env = make_env(arglist.scenario, arglist, arglist.benchmark)
            # Create agent trainers
            # Initialize
            U.initialize()

            # Load previous results, if necessary
            ddpg_index = 200 + i*100
            spac_index = 1800 + i*100
            # ddpg_index = 25000
            # spac_index = 25000
            arglist.load_dir = arglist.save_dir
            load_dir = arglist.load_dir
            ddpg_load_dir = load_dir + 'policy/' + '-' + '{}'.format(ddpg_index)
            spac_load_dir = load_dir + 'spacbackup/' + 'cps-' + '{}'.format(spac_index)
            print(ddpg_load_dir)
            # print(spac_load_dir)
            # print('Loading previous state...')
            U.load_state(ddpg_load_dir)

            hid_dims = [64,32]
            qf_hid_dims = [64,32]
            odims = [env.observation_space.shape[0], env.observation_space.shape[0]]
            adims = [1, 1]
            spac_trainers = [QFPolicy(i, seed + i, odims, adims, hid_dims, qf_hid_dims) for i in range(2)]
            # spac_trainers[0].load_model()
            spac_trainers[1].load_model(spac_index)
            # trainers[1] = trainers[0]
            trainers[1] = spac_trainers[1]


            for j in range(1):
                done_episode_rewards = [0.0]  # sum of rewards for all agents
                ter_episode_rewards = [0.0]
                agent_rewards = [[0.0] for _ in range(2)]  # individual agent reward
                agent_info = [[[]]]  # placeholder for benchmarking info
                saver = tf.train.Saver(max_to_keep=300)
                # obs_n = env.reset()
                episode_step = 0
                train_step = 0
                t_start = time.time()

                # print('Starting iterations...')
                pong.episode_restart()
                obs_n = [env.reset() for env in participants]
                while True:
                    action_n = [agent.action(obs) for agent, obs in zip(trainers, obs_n)]
                    for a, env in zip(action_n, participants):
                        env.unwrapped.apply_action(a)

                    pong.global_step()

                    done = False
                    state_reward_done_info = [env.step(a) for a, env in zip(action_n, participants)]
                    new_obs_n = [x[0] for x in state_reward_done_info]
                    rew_n = [x[1] for x in state_reward_done_info]
                    if rew_n[1] ==-1:
                        rew_n[0] = 1
                        done = True
                    if rew_n[0] ==-1:
                        rew_n[1] = 1
                        done = True
                    for i in range(len(rew_n)):
                        if rew_n[i]!=-1 and rew_n[i]!=1:
                            if rew_n[i] != 0:
                                rew_n[i] = 0
                    done_n = [x[2] for x in state_reward_done_info]
                    info_n = [x[3] for x in state_reward_done_info]

                    # done = all(done_n)
                    terminal = (episode_step >= arglist.max_episode_len)
                    obs_n = new_obs_n

                    for i, rew in enumerate(rew_n):
                        agent_rewards[i][-1] += rew

                    # if terminal:
                    #     pong.episode_restart()
                    #     obs_n = [env.reset() for env in participants]
                    #     episode_step = 0
                    #     ter_episode_rewards.append(0)
                    #     agent_info.append([[]])
                    if done:
                        pong.episode_restart()
                        obs_n = [env.reset() for env in participants]
                        episode_step += 1
                        agent_info.append([[]])
                    # increment global step counter
                    train_step += 1

                    diff = agent_rewards[1][0] - agent_rewards[0][0]
                    if diff != 0 and episode_step==50 and done:
                        print('diff: {}, index: {}'.format(diff, i))
                    if episode_step > 50:
                        final_ag0_rewards.append(agent_rewards[0][0])
                        final_ag1_rewards.append(agent_rewards[1][0])
                        break

                    # for displaying learned policies
                    if arglist.display:
                    # if True:
                        # time.sleep(0.1)
                        still_open = pong.test_window()
                        # print(np.mean(episode_rewards[-arglist.save_rate:]))
                        continue

                    # update all trainers, if not in display or benchmark mode
                    # loss = None
                    # for agent in trainers:
                    #     agent.preupdate()
                    # for agent in trainers:
                    #     loss = agent.update(trainers, train_step+200)

                        # print(loss)

                    # save model, display training output
                    # if done and (len(done_episode_rewards) % arglist.save_rate == 0):
                    #     if len(done_episode_rewards) > 0 and len(done_episode_rewards) % arglist.save_rate == 0:
                    #         final_done_rewards.append(np.mean(done_episode_rewards[-arglist.save_rate:]))
                    #         final_ter_rewards.append(np.mean(ter_episode_rewards[-arglist.save_rate:]))
                    #         # print(len(episode_rewards))
                    #         rew_file_name = arglist.plots_dir + arglist.exp_name + '_done_rewards.pkl'
                    #         with open(rew_file_name, 'wb') as fp:
                    #             pickle.dump(final_done_rewards, fp)
                    #         agrew_file_name = arglist.plots_dir + arglist.exp_name + '_ter_rewards.pkl'
                    #         with open(agrew_file_name, 'wb') as fp:
                    #             pickle.dump(final_ter_rewards, fp)


                    if terminal and (len(ter_episode_rewards) % arglist.save_rate == 0):
                        print()
                        # U.save_state(arglist.save_dir, saver=saver, step=len(ter_episode_rewards))
                        # # print statement depends on whether or not there are adversaries
                        # if num_adversaries == 0 and loss is not None:
                        #     print(Fore.GREEN + "\nq_loss: {:.5f}, p_loss: {:.5f}, target_q: {:.5f}, target_q_next: {:.5f}, eps:{} ,done_episodes: {}, ter episode: {}, time: {}".format(
                        #         loss[0], loss[1], loss[2], loss[4], len(ter_episode_rewards), np.mean(done_episode_rewards[-arglist.save_rate:]), np.mean(ter_episode_rewards[-arglist.save_rate:]), round(time.time()-t_start, 3)))
                        #     # print(
                        #     #     Fore.GREEN + "\nsteps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(
                        #     #         train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]),
                        #     #         [np.mean(rew[-arglist.save_rate:]) for rew in agent_rewards],
                        #     #         round(time.time() - t_start, 3)))
                        # else:
                        #     print(Fore.GREEN + "\nsteps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(
                        #         train_step, len(ter_episode_rewards), np.mean(ter_episode_rewards[-arglist.save_rate:]),
                        #         [np.mean(rew[-arglist.save_rate:]) for rew in agent_rewards], round(time.time()-t_start, 3)))
                        # t_start = time.time()
                        # Keep track of final episode reward
                        # final_ep_rewards.append(np.mean(ter_episode_rewards[-arglist.save_rate:]))
                        # for rew in agent_rewards:
                        #     final_ep_ag_rewards.append(np.mean(rew[-arglist.save_rate:]))

                    # saves final episode reward for plotting training curve later
                    # if len(episode_rewards) > 0 and len(episode_rewards) % arglist.save_rate == 0:
                    #     # print(len(episode_rewards))
                    #     rew_file_name = arglist.plots_dir + arglist.exp_name + '_rewards.pkl'
                    #     with open(rew_file_name, 'wb') as fp:
                    #         pickle.dump(final_ep_rewards, fp)
                    #     agrew_file_name = arglist.plots_dir + arglist.exp_name + '_agrewards.pkl'
                    #     with open(agrew_file_name, 'wb') as fp:
                    #         pickle.dump(final_ep_ag_rewards, fp)
                        # print('...Finished total of {} episodes.'.format(len(episode_rewards)))
                        # break

                    if arglist.display:
                        if not still_open: break

if __name__ == '__main__':
    arglist = parse_args()
    train(arglist)
