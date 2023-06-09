import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LSTM, Flatten

import gym
import argparse
import numpy as np

# import random
# import tensorflow as tf
import matplotlib.pyplot as plt
# import csv
# import itertools
# import tensorflow.contrib.slim as slim
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import math
import os

import random
import itertools
import scipy.ndimage
import matplotlib.pyplot as plt
#
import dgl
from dgl.nn import GraphConv
from helper import *
from environment import *
from network import *
import datetime

## TODO
## 1. check loss: checked
## 2. modify to batch size: only do not know how to modify VALUE BATCH
## 3. easy task

## 思考：改batch：不大成功，no gradient w.r.t model variables
## 另一个方案：改成batch输入，每次通过model，policy——soft和value都是由model，action记录，predicted model 分开传action与hidden

class ReplayAgent:
    def __init__(self, env):
        self.env = env
        self.input_dim = self.env.state_dim
        self.action_dim = self.env.num_actions
        self.model = A2CMetaNetwork(self.action_dim, self.input_dim, 48)
        self.internal_world = InternalWorld(52)

        # self.model = A2CMetaNetwork(self.action_dim,self.input_dim, 48)
        self.opt = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
        self.replay_max = 8

    def list_to_batch(self, list):
        batch = list[0]
        for elem in list[1:]:
            batch = np.append(batch, elem, axis=0)
        return batch

    def replay(self, state, predicted_goal, predicted_location, agent_state, action, time):
        n = 0

        done = False
        augment_length = self.replay_max * self.action_dim + 1
        augment = np.zeros(augment_length)
        output = np.zeros(augment_length)
        # initialize predicted goal and location
        max_goal = np.argmax(predicted_goal)
        # take action a0, and next state s1 is max_state
        max_state = np.argmax(predicted_location)
        action_onehot = tf.one_hot(action, self.action_dim, dtype=tf.float32)
        output[0:4] = action_onehot
        n += 1
        time += 400

        # check if reach goal
        if max_goal == max_state:
            done = True
        else:
            reward = 0
        # at state, take action and we are at predicted location

        while n < self.replay_max and not done:
            action_soft, action, value, agent_state, _, predicted_location, _ = self.model(max_state, reward,
                                                                                           action_onehot, time / 20000,
                                                                                           self.env.wall, augment,
                                                                                           agent_state)
            max_state = np.argmax(predicted_location)
            n += 1
            action_onehot = tf.one_hot(action, self.action_dim, dtype=tf.float32)
            ind = n * 4
            output[ind:(ind + 4)] = action_onehot
            time += 400
            if max_goal == max_state:
                done = True
                reward = 1
            else:
                reward = 0
        if time > 20000:  # if time exceed, done is false whatever
            done = False
        output[-1] = np.array([done]).astype(int)[0]
        return output

    def list_to_batch(self, list):
        batch = list[0]
        for elem in list[1:]:
            batch = np.append(batch, elem, axis=0)
        return batch

    def train(self, max_episodes=201):
        print("Start training")
        # print("environment wall",self.env.wall)
        # print("environment starting position",self.env.hero.x,self.env.hero.y)
        # print("environment starting position",self.env.goal.x,self.env.goal.y)


        starttime = datetime.datetime.now()
        total_reward = []
        total_losses = []
        for ep in range(max_episodes):

            print("Episode", ep)
            state_batch = []
            action_batch = []
            reward_batch = []
            time_batch = []
            augment_batch = []
            agent_state_batch = []
            predicted_location_batch = []
            predicted_goal_batch = []
            true_goal_batch = []
            action_soft_batch = []
            value_batch = []


            done = False
            trial_done = False
            episode_reward = [0, 0, 0, 0]
            episode_frames = []  # store images
            episode_rewards = 0

            ## reset state of env and agent hidden state at the beginning of episode
            # reward_color = [np.random.uniform(),np.random.uniform(),np.random.uniform()]
            # state = self.env.reset() # draw an environment from distribution
            state = self.env.reset()  # draw an environment from distribution
            agent_state = self.model.reset(1)  # reset and return [[[0]],[[0]]]

            goal_ind = self.env.goal.x * 4 + self.env.goal.y
            goal_onehot = tf.one_hot(goal_ind, self.env.state_dim, dtype=tf.float32)

            ## initialize action
            action = -1  # no action at t=-1
            reward = 0
            timestep = 0
            time = 0
            augment_length = self.replay_max * self.action_dim + 1
            augment = np.zeros(augment_length)
            state = tf.cast(np.reshape(state, [1, state.shape[0]]),tf.float32)
            action = np.reshape(action, [1])
            reward = tf.cast(np.reshape(reward, [1, 1]),tf.float32)
            timestep = tf.cast(np.reshape(timestep, [1, 1]),tf.float32)
            time = tf.cast(np.reshape(time, [1, 1]),tf.float32)
            augment = tf.cast(np.reshape(augment, [1, augment.shape[0]]),tf.float32)
            goal_onehot = tf.cast(np.reshape(goal_onehot, [1,16]),tf.float32)

            time_batch.append(time)
            state_batch.append(state)
            action_batch.append(action) # 1
            reward_batch.append(reward) # 1,1
            augment_batch.append(augment) #
            agent_state_batch.append(agent_state)
            true_goal_batch.append(goal_onehot)
            action_soft_batch.append(tf.one_hot(action, depth=4)) # 1,4


            while not done:
                trial_done = False
                while (not trial_done) and (not done):

                    if action == -1:
                        action_onehot = tf.constant(np.zeros([1, 4]), dtype=tf.float32)
                    else:
                        action_onehot = tf.one_hot(action, self.action_dim, dtype=tf.float32)

                    # forward pass
                    # action_soft, action, value, agent_state, predicted_goal, predicted_location, rollout = self.model(
                    #     state, reward, action_onehot, time / 20000, self.env.wall, augment, agent_state)
                    wall_feat, wall_adj = preprocess_wall(self.env.wall, num=1)
                    action_soft, value, agent_state_h, agent_state_c, whole = self.model(state, reward, action_onehot, time / 20000, wall_feat, wall_adj, augment, agent_state)

                    action_soft_r = np.array(action_soft).astype('float64')
                    action_soft_r = action_soft_r / np.sum(action_soft_r)
                    action = np.random.choice(self.action_dim, p=action_soft_r[0])
                    action_onehot = tf.one_hot([action] , self.action_dim ,dtype=tf.float32) # 1,4

                    predicted_goal, predicted_location, rollout = self.internal_world(whole, action_onehot)
                    agent_state = [agent_state_h, agent_state_c]
                    # action_soft [batch,1,4], value [batch,1,1], agent_state [batch, 1, units], predicted_goal[batch, 1,16], rollout[batch,1,1]

                    if rollout[0, 0] > 1:
                        print("rollout")
                        augment = self.replay(state, predicted_goal, predicted_location, agent_state, action, time)
                        time = self.env.replay()  # update time, not timestep and state
                        continue

                    else:
                        next_state, reward, done, trial_done, timestep, time = self.env.step(action)  # grid
                        action = np.reshape(action, [1])
                        next_state = tf.cast(np.reshape(next_state, [1, next_state.shape[0]]),tf.float32) # for gridworld
                        reward = tf.cast(np.reshape(reward, [1, 1]),tf.float32)
                        timestep = tf.cast(np.reshape(timestep, [1, 1]),tf.float32)
                        time = tf.cast(np.reshape(time, [1, 1]),tf.float32)
                        augment = tf.cast(np.zeros([1, augment_length]),tf.float32)
                        state = next_state

                        # update batch record
                        time_batch.append(time)
                        state_batch.append(state)
                        action_batch.append(action)
                        reward_batch.append(reward)
                        augment_batch.append(augment)
                        agent_state_batch.append(agent_state)
                        episode_reward[action[0]] += reward[0][0].numpy()
                        predicted_location_batch.append(predicted_location)
                        predicted_goal_batch.append(predicted_goal)
                        true_goal_batch.append(true_goal_batch[-1]) # same goal
                        action_soft_batch.append(action_soft)
                        value_batch.append(value)

                        if ep % 10 == 0:
                            # reward and frame update
                            # episode_frames.append(set_image_bandit(episode_reward,self.env.bandit,action[0],timestep[0][0]))
                            frame = set_image_mazeworld(self.env.renderAll(), episode_reward, timestep[0][0].numpy(),ep)
                            episode_frames.append(frame)

                # after trial done, but not done in episode
                if done == False:
                    # feed the reached location to network and ignore action
                    action_onehot = tf.one_hot(action, self.action_dim, dtype=tf.float32)
                    time += 400
                    self.env.time = time[0][0]
                    # print(self.env.time)
                    # wall: 4x4
                    wall_feat, wall_adj = preprocess_wall(self.env.wall, num=1)
                    action_soft, value, agent_state_h, agent_state_c, whole = self.model(state, reward, action_onehot, time / 20000,
                                                                 wall_feat, wall_adj, augment, agent_state)
                    agent_state = [agent_state_h, agent_state_c]

                    # action_soft_r = np.array(action_soft).astype('float64')
                    # action_soft_r = action_soft_r / np.sum(action_soft_r)
                    # action = np.random.choice(4, p=action_soft_r[0])
                    # action_onehot = tf.one_hot([action], self.num_actions, dtype=tf.float32)  # 1,4
                    #
                    # predicted_goal, predicted_location, rollout = self.internal_world(agent_state, action_onehot)
                    # action_soft [batch,1,4], value [batch,1,1], agent_state [batch, 1, units], predicted_goal[batch, 1,16], rollout[batch,1,1]

                    # value, predicted_goal, agent_state -> useful
                    # value state
                    value_batch.append(value)


                    # reset state after each trial
                    state = self.env.reset_trial()

                    goal_ind = self.env.goal.x * 4 + self.env.goal.y
                    goal_onehot = tf.one_hot(goal_ind, self.env.state_dim, dtype=tf.float32)

                    action = -1  #
                    reward = 0  #
                    augment_length = self.replay_max * self.action_dim + 1
                    augment = np.zeros(augment_length)

                    # reshape
                    state = tf.cast(np.reshape(state, [1, state.shape[0]]),tf.float32)  # for gridworld
                    action = np.reshape(action, [1])
                    reward = tf.cast(np.reshape(reward, [1, 1]),tf.float32)
                    augment = tf.cast(np.reshape(augment, [1, augment.shape[0]]),tf.float32)
                    goal_onehot = tf.cast(np.reshape(goal_onehot, [1,16]),tf.float32)

                    time_batch.append(time)
                    state_batch.append(state)
                    action_batch.append(action)
                    reward_batch.append(reward)
                    augment_batch.append(augment)
                    predicted_location_batch.append(state)  # at the beginning, set the same prediction with true
                    predicted_goal_batch.append(goal_onehot)
                    true_goal_batch.append(goal_onehot)
                    agent_state_batch.append(agent_state)
                    action_soft_batch.append(tf.one_hot(action,depth=4))

            discounted_reward = cal_discounted_reward(reward_batch, gamma)
            # transform variable
            action_onehot_batch = tf.one_hot(action_batch, depth=4)  # 52,1,4
            action_onehot_batch = np.reshape(action_onehot_batch,
                                             [len(action_onehot_batch), 1, 4])
            tg_targets = tf.convert_to_tensor(discounted_reward)
            tg_targets = np.reshape(tg_targets, [tg_targets.shape[0], 1, 1])
            state_batch = np.reshape(state_batch, [len(state_batch), 1, 16])
            true_goal_batch = np.reshape(true_goal_batch, [len(true_goal_batch), 1, 16])

            print("tg_targets", tg_targets)
            # print("value", value_batch)
            print("statebatch", state_batch)
            # print("predicted location", predicted_location_batch)  # 51,1,16
            print("true goal", true_goal_batch)
            # print("predicted goal", predicted_goal_batch)
            print("action_soft", action_soft_batch)  # action soft batch: [51,1,4]
            print("discount", discounted_reward)
            print("action one hot", action_onehot_batch)  # 51,1,4
            print("model variables", self.model.variables)

            if train == True:  # train every episode
                with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
                    tape1.watch(self.model.variables)
                    tape2.watch(self.internal_world.variables)
                    print("backpropagation")
                    game_length = len(state_batch) - 1
                    # assert game_length == timestep
                    agent_state = self.model.reset(1)  # reset
                    total_loss = 0
                    internal_total_loss = 0
                    total_total_loss = 0
                    for i in range(game_length):

                        # action_soft, action, value, agent_state, predicted_goal, predicted_location, rollout = self.model(
                        #             state_batch[i], reward_batch[i], action_onehot_batch[i], time_batch[i] / 20000, self.env.wall,
                        #             augment_batch[i], agent_state)
                        wall_feat, wall_adj = preprocess_wall(self.env.wall, num=1)
                        action_soft, value, agent_state_h, agent_state_c, whole = self.model(state_batch[i], reward_batch[i], action_onehot_batch[i], time_batch[i] / 20000,
                                                                     wall_feat, wall_adj, augment_batch[i], agent_state)

                        # action_soft_r = np.array(action_soft).astype('float64')
                        # action_soft_r = action_soft_r / np.sum(action_soft_r)
                        # action = np.random.choice(4, p=action_soft_r[0])
                        # action_onehot = tf.one_hot([action], self.num_actions, dtype=tf.float32)  # 1,4

                        predicted_goal, predicted_location, rollout = self.internal_world(whole, action_onehot_batch[i+1])
                        # action_soft [batch,1,4], value [batch,1,1], agent_state [batch, 1, units], predicted_goal[batch, 1,16], rollout[batch,1,1]
                        agent_state = [agent_state_h, agent_state_c]

                        # value loss & policy loss
                        loss_fn = tf.keras.losses.MeanSquaredError()
                        value_loss = loss_fn(tg_targets[i+1], value)
                        # print("vale gradient",tf.gradients(value_loss,self.model.variables))
                        # print(value_loss)


                        advantage = tg_targets[i+1]- value  # check!
                        # print("advantage",advantage)

                        ce_loss = tf.keras.losses.CategoricalCrossentropy()
                        policy_loss = ce_loss(action_onehot_batch[i+1], action_soft,
                                              sample_weight=advantage)  # current action
                        print(policy_loss)

                        # print("polic gradient",tf.gradients(policy_loss,self.model.variables))
                        ## entropy loss
                        entropy_loss = tf.reduce_sum(action_soft * tf.math.log(action_soft))
                        ## internal world loss
                        ce_loss = tf.keras.losses.CategoricalCrossentropy() #50,1,16 vs 50,1,16
                        internal_loss = ce_loss(state_batch[i+1], predicted_location) + ce_loss(true_goal_batch[i+1], predicted_goal)

                        total_loss += policy_loss - entropy_loss * 0.05 + value_loss * 0.05
                        internal_total_loss += internal_loss * 0.5
                        # total_loss += policy_loss

# check gradient**
                    total_loss /= game_length
                    internal_total_loss /= game_length
                    grad = tape1.gradient(total_loss, self.model.variables)
                    internal_grad = tape2.gradient(internal_total_loss, self.internal_world.variables)
                    grad, _ = tf.clip_by_global_norm(grad, 50.0)
                    internal_grad, _ = tf.clip_by_global_norm(internal_grad, 50.0)
                    print("gradient",grad)
                    print("internal gradient",internal_grad)



                    # chain variables and gradient together
                    print("model variables",self.model.variables)
                    print("internal model variables",self.internal_world.variables)

                    total_grad = grad + internal_grad
                    print("total grad", total_grad)

                    total_variables = self.model.variables + self.internal_world.variables

                    self.opt.apply_gradients(list(zip(total_grad, total_variables)))
                    # self.opt.apply_gradients(list(zip(internal_grad, self.internal_world.variables)))
                total_total_loss = total_loss+internal_total_loss
                total_losses.append(total_total_loss)
            episode_rewards = np.sum(episode_reward)
            print('EP{} EpisodeReward={}'.format(ep, episode_rewards))
            total_reward.append(episode_rewards)
            # wandb.log({'Reward': episode_rewards})

            # if episode==100, make a gif and save model
            if ep % 10 == 0:
                self.images = np.array(episode_frames)
                make_gif(self.images, './frames/image' + str(ep) + '.gif',
                         duration=len(self.images) * 0.1, true_image=True)

                if train == True:
                    print("save!")
                    #   # save model actor
                    checkpoint_path = "./model/metaRL" + str(ep)
                    self.model.save_weights(checkpoint_path)
                    checkpoint_path2 = "./model/internal" + str(ep)
                    self.internal_world.save_weights(checkpoint_path2)
        endtime = datetime.datetime.now()
        print((endtime - starttime).seconds,'seconds')
        plt.figure()
        plt.plot(total_reward)
        plt.show()
        plt.savefig("reward_loss/reward_single.jpg")
        plt.figure()
        plt.plot(total_losses)
        plt.savefig("reward_loss/loss_single.jpg")

if __name__ == '__main__':
    gamma = 0.8 # discount
    # update_interval = 5
    # actor_lr = 0.0005
    # critic_lr = 0.001
    tf.keras.backend.set_floatx('float32')
    learning_rate = 0.001
    load_model = False
    train = True
    actor_model_path = "./model/actor0"
    critic_model_path = "./model/critic0"
    meta_model_path = './model/metaRL0'
    #
    # os.environ['WANDB_API_KEY'] = 'a8f58ffcd8fd97f1ed56be1bb36e1190ff6731d6'
    # wandb.login()
    #
    # wandb.init(name='A2C-replay', project="replay-RL")

    agent = ReplayAgent(mazeworld(size=4))

    # f = open("output.txt", "a")


    agent.train()
    print("finish!!!")
    # f.close()

