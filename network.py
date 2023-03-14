import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LSTMCell, Flatten
import numpy as np
import dgl
from dgl.nn import GraphConv

class A2CMetaNetwork(tf.keras.Model):
    def __init__(self, num_actions, input_dim, units):
        super().__init__()
        self.input_dim = input_dim
        self.num_actions = num_actions
        self.input_layer = Input(self.input_dim)
        # embedding
        self.conv = Dense(64, activation=tf.keras.activations.elu)
        ############################TODO#######################
        # graph embedding
        self.gcn = GraphConv(16, 4, norm='both', weight=True, bias=True)
        # input: graph, feature
        # output: node * output feature dimension
        self.flatten = Flatten()

        # augment embedding

        #######################################################
        # lstm
        self.units = units
        self.lstm_core = LSTMCell(self.units)
        # output two model network
        self.policy_layer = Dense(self.num_actions,
                                  activation=tf.keras.activations.softmax)
        self.value_layer = Dense(1,activation='linear')
        # output predictive model
        self.internal_world = Dense(33, activation=tf.keras.activations.relu)
        self.predict_goal = Dense(16, activation=tf.keras.activations.softmax)
        self.predict_location = Dense(16, activation=tf.keras.activations.softmax)

        # output rollout
        self.rollout = Dense(1, activation=tf.keras.activations.sigmoid)

    def call(self, state, reward, action_onehot, time, wall, augment ,hidden):
        # obs = self.input_layer(obs)
        # obs = self.flatten(state)
        obs = self.conv(state) # 64
        # obs = tf.cast(obs, tf.float32)

        # encode wall
        # if only one batch

        ## runable
        # src, dst = np.nonzero(wall)
        # g = dgl.graph((src, dst))
        # g = dgl.add_self_loop(g)
        # # feat = torch.from_numpy(np.ones([16 ,16])).float()
        # feat = tf.ones([16,16],tf.float32)
        # wall_embed = self.gcn(g, feat)
        # # wall_embed = self.flatten(tf.convert_to_tensor([wall_embed]))
        # print(wall_embed)
        # wall_embed = self.flatten(wall_embed)
        # wall_embeds = tf.repeat(wall_embed, repeats=state.shape[0],axis=0)


        # wall_embed = tf.convert_to_tensor([wall_embed.detach().numpy()])

        # print(wall_embed)
        # wall_embeds = tf.convert_to_tensor(wall_embeds)
        # wall_embeds = tf.cast(wall_embeds, tf.float32)

        # encode augment
        # print(obs, reward, action_onehot, time, wall_embed, augment)
        wall_embeds = tf.zeros([state.shape[0],64])


        # concatenate input
        input = tf.concat([obs, reward, action_onehot, time, wall_embeds, augment] ,1  )# for gridworld 64+1+4+1+64+33 =
        # input = tf.convert_to_tensor([input])

        # lstm network
        # hidden = tf.convert_to_tensor(hidden)
        # hidden = tf.cast(hidden, dtype=tf.float32)

        out, agent_state = self.lstm_core(input, hidden) # 1, 167 input
        # output: 1, units
        # out = tf.cast(out ,tf.float32)


        # agent_state = [agent_state_c ,agent_state_h]
        action_soft = self.policy_layer(out)
        # action_soft = tf.cast(action_soft, tf.float63
        action_soft = action_soft / tf.reduce_sum(action_soft) # 1.16
        action_soft_r = np.array(action_soft).astype('float64')
        action_soft_r = action_soft_r / np.sum(action_soft_r)

        # print(action_soft)

        value = self.value_layer(out)
        action = np.random.choice(self.num_actions, p=action_soft_r[0])
        action_onehot = tf.one_hot([action] ,self.num_actions ,dtype=tf.float32) # 1,4

        # internal model
        internal_input = tf.concat([out, action_onehot] ,1)
        internal_hidden = self.internal_world(internal_input)
        predicted_goal = self.predict_goal(internal_hidden)
        predicted_location = self.predict_location(internal_hidden)

        # rollout
        rollout = self.rollout(out)

        return action_soft, action, value, agent_state, predicted_goal, predicted_location, rollout
    # [1,16], 1, [1,1], [1,64], [1,16],[1,16],[1,1]


    def reset(self,batch_size):
        return [tf.zeros((batch_size,self.units)), tf.zeros((batch_size,self.units))]