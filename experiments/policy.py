"""
"""
import numpy as np
import tensorflow as tf
from baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer

class QFPolicy(object):
    def __init__(self, id, seed, odims, adims, hid_dims, qf_hid_dims,
                 max_pool_size=int(1e6), p_lr = 2e-3, q_lr = 3e-3,te = 1e-2,
                 ): # p_lr = 1e-3, q_lr = 5e-3
        self.id = id
        self.seed = seed
        self.odims = odims
        self.adims = adims
        self.adim = adims[id]
        self.n = len(odims)
        self.hid_dims = hid_dims
        self.qf_hid_dims = qf_hid_dims
        self.p_lr = p_lr
        self.q_lr = q_lr
        self.te = te
        self.pool = ReplayBuffer(max_pool_size)
        self._build_graph()
        self._init_session()

    def _build_graph(self):
        self.g = tf.Graph()
        with self.g.as_default():
            self._placeholders()
            self._build_nn()
            self._loss_train_op()
            self.init = tf.global_variables_initializer()

    def _init_session(self):
        self.sess = tf.Session(graph=self.g)
        self.sess.run(self.init)
        # self.a_v.set_session(self.sess)
        # self.q_v.set_session(self.sess)
        # self.a__v.set_session(self.sess)
        # self.q__v.set_session(self.sess)

    def _placeholders(self):
        o_sm_n = []
        for i in range(self.n):
            o_sm_n.append(tf.placeholder(tf.float32, (None, self.odims[i]), "observation"+str(i)))
        self.o_sm = o_sm_n[self.id]
        self.o_sm_n = tf.concat(o_sm_n, 1)

        a_sm_n = []
        for i in range(self.n):
            a_sm_n.append(tf.placeholder(tf.float32, (None, self.adims[i]), "action"+str(i)))
        self.a_sm_n = tf.concat(a_sm_n, 1)
        self.a_sm_list = a_sm_n

        self.q_target_sm = tf.placeholder(tf.float32, [None, 1], name="target")

    def _build_nn(self):
        def policy_nn(s, scope, trainable, reuse=False):
            tf.set_random_seed(self.seed)
            with tf.variable_scope(scope, reuse=reuse):
                h = s
                for i, n in enumerate(self.hid_dims):
                    h = tf.layers.dense(h, n, activation=tf.nn.relu,
                                         name="polfc%i"%(i+1), trainable=trainable)
                # logits=tf.layers.dense(h, self.adim, name='pfinal', trainable=trainable)
                logits = tf.layers.dense(h, 1, name='pfinal', trainable=trainable)
                # u = tf.random_uniform(tf.shape(logits))
                # action = tf.nn.softmax(logits - tf.log(-tf.log(u)), axis=-1)
                return logits

        def qf_nn(s, a, scope, trainable, reuse=False):
            tf.set_random_seed(self.seed)
            with tf.variable_scope(scope, reuse=reuse):
                h = tf.concat([s, a], axis=1, name='input')
                for i, n in enumerate(self.qf_hid_dims):
                    h = tf.layers.dense(h, n, activation=tf.nn.relu,
                                        name="qf%i"%(i+1), trainable=trainable)
                q = tf.layers.dense(h, 1, name='qfinal', trainable=trainable)
                return q

        self.a = policy_nn(self.o_sm, 'p_eval', True)

        self.q = qf_nn(self.o_sm_n, self.a_sm_n, 'q_eval', True)

        self.a_sm_list[self.id] = self.a
        self.a_sm_n_a = tf.concat(self.a_sm_list, 1)
        self.qa = qf_nn(self.o_sm_n, self.a_sm_n_a, 'q_eval', False, reuse=True)

        self.a_ = policy_nn(self.o_sm, 'p_target', False)
        self.q_ = qf_nn(self.o_sm_n, self.a_sm_n, 'q_target', False)

        self.pe_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='p_eval')
        self.pt_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='p_target')
        self.qe_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_eval')
        self.qt_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_target')

        # target net replacement
        self.soft_replace = [[tf.assign(pt, (1-self.te)*pt + self.te*pe),
                              tf.assign(qt, (1-self.te)*qt + self.te*qe)]
                             for pt, pe, qt, qe in zip(self.pt_params, self.pe_params,
                                                       self.qt_params, self.qe_params)]
        # self.a_v = TFVariables(self.a)
        # self.q_v = TFVariables(self.q)
        # self.a__v = TFVariables(self.a_)
        # self.q__v = TFVariables(self.q_)

        self.a_v = self.a
        self.q_v = self.q
        self.a__v = self.a_
        self.q__v = self.q_
        self.saver = tf.train.Saver()

    def _loss_train_op(self):
        td_error = tf.reduce_mean(tf.square(self.q - self.q_target_sm))
        qf_optimizer = tf.train.AdamOptimizer(self.q_lr)
        self.q_train_op = qf_optimizer.minimize(td_error, var_list=self.qe_params) #

        policy_loss = -tf.reduce_mean(self.qa)
        p_optimizer = tf.train.AdamOptimizer(self.p_lr)
        self.p_train_op = p_optimizer.minimize(policy_loss, var_list=self.pe_params) #

    def set_policy_params(self, a_v, a__v):
        self.a_v.set_flat(a_v)
        self.a__v.set_flat(a__v)

    def get_policy_params(self):
        a_v = self.a_v.get_flat()
        a__v = self.a__v.get_flat()
        return a_v, a__v

    def action(self, obs):
        if np.ndim(obs) == 1:
            obs=obs[np.newaxis,:]
        feed_dict = {self.o_sm: obs}
        a = self.sess.run(self.a, feed_dict=feed_dict)
        return a[0]

    def get_target_action(self, obs):
        if np.ndim(obs) == 1:
            obs=obs[np.newaxis,:]
        feed_dict = {self.o_sm: obs}
        a = self.sess.run(self.a_, feed_dict=feed_dict)
        return a

    def get_target_q(self, obs, act):
        if np.ndim(obs) == 1:
            obs=obs[np.newaxis,:]
        feed_dict = {self.a_sm_n: act,
                     self.o_sm_n: obs}
        q_ = self.sess.run(self.q_, feed_dict=feed_dict)
        return q_

    def experience(self, obs, act, rew, new_obs, done):
        # Store transition in the replay buffer.
        # self.pool.add(obs, act, rew, new_obs, float(done))
        self.pool.add(obs, act, rew, new_obs, done)

    def pq_soft_replace(self):
        self.sess.run(self.soft_replace)

    def q_train(self, obs, acts, q_target):
        feed_dict_q = {self.o_sm_n: obs,
                       self.a_sm_n: acts,
                       self.q_target_sm: q_target,}
        self.sess.run(self.q_train_op, feed_dict_q)

    def p_train(self, obs, obs_n, act_n):
        feed_dict_p = {self.o_sm_n: obs,
                       self.o_sm: obs_n[self.id],}
        for i in range(len(act_n)):
            if not i==self.id: feed_dict_p.update({self.a_sm_list[i]: act_n[i]})

        self.sess.run(self.p_train_op, feed_dict_p)

    def save_model(self):
        checkpoint = '/home/lsq/PycharmProjects/spac/spacbackup/cps'
        self.saver.save(self.sess, checkpoint)

    def load_model(self, index):
        load_dir = '/home/lsq/PycharmProjects/spac/spacbackup/'
        spac_load_dir = load_dir  + '0cps-' + '{}'.format(index)
        print(spac_load_dir)
        self.saver.restore(self.sess, spac_load_dir)

    def close_sess(self):
        self.sess.close()


class QFDGPolicy(object):
    def __init__(self, id, seed, odims, adims, hid_dims, qf_hid_dims,
                 max_pool_size=1e6, p_lr = 9e-4, q_lr = 8e-3,te = 1e-2,
                 ):
        self.id = id
        self.seed = seed
        self.odims = odims
        self.adims = adims
        self.adim = adims[id]
        self.n = len(odims)
        self.hid_dims = hid_dims
        self.qf_hid_dims = qf_hid_dims
        self.p_lr = p_lr
        self.q_lr = q_lr
        self.te = te
        self.var = 5
        self.pool = ReplayBuffer(max_pool_size)
        self._build_graph()
        self._init_session()

    def _build_graph(self):
        self.g = tf.Graph()
        with self.g.as_default():
            self._placeholders()
            self._build_nn()
            self._loss_train_op()
            self.init = tf.global_variables_initializer()

    def _placeholders(self):
        o_sm_n = []
        for i in range(self.n):
            o_sm_n.append(tf.placeholder(tf.float32, (None, self.odims[i]), "observation"+str(i)))
        self.o_sm = o_sm_n[self.id]
        self.o_sm_n = tf.concat(o_sm_n, 1)

        a_sm_n = []
        for i in range(self.n):
            a_sm_n.append(tf.placeholder(tf.float32, (None, self.adims[i]), "action"+str(i)))
        self.a_sm_n = tf.concat(a_sm_n, 1)
        self.a_sm_list = a_sm_n

        self.q_target_sm = tf.placeholder(tf.float32, [None, 1], name="target")

    def _build_nn(self):
        def policy_nn(s, scope, trainable, reuse=False):
            tf.set_random_seed(self.seed)
            with tf.variable_scope(scope, reuse=reuse):
                h = s
                for i, n in enumerate(self.hid_dims):
                    h = tf.layers.dense(h, n, activation=tf.nn.relu,
                                         name="polfc%i"%(i+1), trainable=trainable)
                action=tf.layers.dense(h, self.adim, name='pfinal',activation=tf.nn.tanh, trainable=trainable)
                return action

        def qf_nn(s, a, scope, trainable, reuse=False):
            tf.set_random_seed(self.seed)
            with tf.variable_scope(scope, reuse=reuse):
                h = tf.concat([s, a], axis=1, name='input')
                for i, n in enumerate(self.qf_hid_dims):
                    h = tf.layers.dense(h, n, activation=tf.nn.relu,
                                        name="qf%i"%(i+1), trainable=trainable)
                q = tf.layers.dense(h, 1, name='qfinal', trainable=trainable)
                return q

        self.a = policy_nn(self.o_sm, 'p_eval', True)

        self.q = qf_nn(self.o_sm_n, self.a_sm_n, 'q_eval', True)

        self.a_sm_list[self.id] = self.a
        self.a_sm_n_a = tf.concat(self.a_sm_list, 1)
        self.qa = qf_nn(self.o_sm_n, self.a_sm_n_a, 'q_eval', False, reuse=True)

        self.a_ = policy_nn(self.o_sm, 'p_target', False)
        self.q_ = qf_nn(self.o_sm_n, self.a_sm_n, 'q_target', False)

        self.pe_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='p_eval')
        self.pt_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='p_target')
        self.qe_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_eval')
        self.qt_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_target')

        # target net replacement
        self.soft_replace = [[tf.assign(pt, (1-self.te)*pt + self.te*pe),
                              tf.assign(qt, (1-self.te)*qt + self.te*qe)]
                             for pt, pe, qt, qe in zip(self.pt_params, self.pe_params,
                                                       self.qt_params, self.qe_params)]

    def _loss_train_op(self):
        td_error = tf.reduce_mean(tf.square(self.q - self.q_target_sm))
        qf_optimizer = tf.train.AdamOptimizer(self.q_lr)
        self.q_train_op = qf_optimizer.minimize(td_error, var_list=self.qe_params) #

        policy_loss = -tf.reduce_mean(self.qa)
        p_optimizer = tf.train.AdamOptimizer(self.p_lr)
        self.p_train_op = p_optimizer.minimize(policy_loss, var_list=self.pe_params) #

    def _init_session(self):
        self.sess = tf.Session(graph=self.g)
        self.sess.run(self.init)

    def get_action(self, obs):
        if np.ndim(obs) == 1:
            obs=obs[np.newaxis,:]
        feed_dict = {self.o_sm: obs}
        a = self.sess.run(self.a, feed_dict=feed_dict)
        a = np.random.normal(a[0], self.var)
        return a

    def get_target_action(self, obs):
        if np.ndim(obs) == 1:
            obs=obs[np.newaxis,:]
        feed_dict = {self.o_sm: obs}
        a = self.sess.run(self.a_, feed_dict=feed_dict)
        return a

    def get_target_q(self, obs, act):
        if np.ndim(obs) == 1:
            obs=obs[np.newaxis,:]
        feed_dict = {self.a_sm_n: act,
                     self.o_sm_n: obs}
        q_ = self.sess.run(self.q_, feed_dict=feed_dict)
        return q_

    def experience(self, obs, act, rew, new_obs, done):
        # Store transition in the replay buffer.
        self.pool.add(obs, act, rew, new_obs, float(done))

    def pq_soft_replace(self):
        self.sess.run(self.soft_replace)

    def q_train(self, obs, acts, q_target):
        feed_dict_q = {self.o_sm_n: obs,
                       self.a_sm_n: acts,
                       self.q_target_sm: q_target,}
        self.sess.run(self.q_train_op, feed_dict_q)

    def p_train(self, obs, obs_n, act_n):
        feed_dict_p = {self.o_sm_n: obs,
                       self.o_sm: obs_n[self.id],}
        for i in range(len(act_n)):
            if not i==self.id: feed_dict_p.update({self.a_sm_list[i]: act_n[i]})
        self.var *= 0.999
        self.sess.run(self.p_train_op, feed_dict_p)

    def close_sess(self):
        self.sess.close()
