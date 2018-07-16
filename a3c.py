from __future__ import print_function
from collections import namedtuple
import numpy as np
import tensorflow as tf
from model import LSTMPolicy
import six.moves.queue as queue
import scipy.signal
import threading
import distutils.version
from constants import constants
from collections import deque
use_tf12_api = distutils.version.LooseVersion(tf.VERSION) >= distutils.version.LooseVersion('0.12.0')
state_q = deque(maxlen=4)
state_ar = np.zeros((42,42,4),dtype=np.float32)
def discount_reward(reward,gamma,index):
    sh = reward.shape[0]
    
    dum = np.zeros(sh)
    temp = np.copy(index)
    
    for i in range(sh-1):

        power = np.power(gamma,temp[i:sh])                
        dum[i] = np.sum(power*reward[i:sh])
        #if i != sh-1:
        temp = index - index[i+1]
    power = np.power(gamma,temp[sh-1]) 
    dum[sh-1] = np.sum(power*reward[sh-1])
    return dum
def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

def process_rollout(rollout, gamma, lambda_=1.0):
    """
given a rollout, compute its returns and the advantage
"""
    batch_si = np.asarray(rollout.states)
    batch_a = np.asarray(rollout.actions)
    rewards = np.asarray(rollout.rewards)
    vpred_t = np.asarray(rollout.values + [rollout.r])
    rep = np.asarray(rollout.rep)    

    re = np.argmax(rep,axis=1) + 1
    rewards_plus_v = np.asarray(rollout.rewards + [rollout.r])   
    sha = rewards_plus_v.shape[0]
    dum = np.zeros(sha,dtype=np.int32)
    dum[0] = 0
    dum[1] = re[0]
  
    if start_rep <= constants['TRAIN_REP']:
        batch_r = discount(rewards_plus_v, gamma)[:-1]
        delta_t = rewards + gamma * vpred_t[1:] - vpred_t[:-1]
        batch_adv = discount(delta_t, gamma * lambda_)
    else:
        for i in range(1,sha-1):
        	t = np.sum(re[0:i+1])    
        	dum[i+1] = t

        batch_r = discount_reward(rewards_plus_v, gamma,dum)[:-1] #calculating the N-step return 
                                                                  # taking into account the reptition
                                                                  #where N is the decision steps.
        delta_t = rewards + np.power(gamma,re) * vpred_t[1:] - vpred_t[:-1]
        batch_adv = discount_reward(delta_t,gamma,dum)  # calculating the advantage according to the  
                                                        # reptition.
          
        
    # this formula for the advantage comes "Generalized Advantage Estimation":
    # https://arxiv.org/abs/1506.02438
    
    
    features = rollout.features[0]
    return Batch(batch_si, batch_a, batch_adv, batch_r, rollout.terminal, features,rep)

Batch = namedtuple("Batch", ["si", "a", "adv", "r", "terminal", "features","rep"])

class PartialRollout(object):
    """
a piece of a complete rollout.  We run our agent, and process its experience
once it has processed enough steps.
"""
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.r = 0.0
        self.terminal = False
        self.features = []
        self.rep = []

    def add(self, state, action, reward, value, terminal, features,rep):
        self.states += [state]
        self.actions += [action]
        self.rewards += [reward]
        self.values += [value]
        self.terminal = terminal
        self.features += [features]        
        self.rep += [rep]

    def extend(self, other):
        assert not self.terminal
        self.states.extend(other.states)
        self.actions.extend(other.actions)
        self.rewards.extend(other.rewards)
        self.values.extend(other.values)
        self.r = other.r
        self.terminal = other.terminal
        self.features.extend(other.features)        
        self.rep.extend(other.rep)


class RunnerThread(threading.Thread):
    """
One of the key distinctions between a normal environment and a universe environment
is that a universe environment is _real time_.  This means that there should be a thread
that would constantly interact with the environment and tell it what to do.  This thread is here.
""" 
    def __init__(self, env, policy, num_local_steps, visualise,log):
        threading.Thread.__init__(self)

        self.queue = queue.Queue(5)
        self.num_local_steps = num_local_steps
        self.env = env
        self.last_features = None
        self.policy = policy
        self.daemon = True
        self.sess = None
        self.summary_writer = None
        self.visualise = visualise
        self.log = log

    def start_runner(self, sess, summary_writer):
        self.sess = sess
        self.summary_writer = summary_writer
        self.start()

    def run(self):
        with self.sess.as_default():
            self._run()

    def _run(self):
        rollout_provider = env_runner(self.env, self.policy, self.num_local_steps, self.summary_writer, self.visualise,self.log)
        while True:
            # the timeout variable exists because apparently, if one worker dies, the other workers
            # won't die with it, unless the timeout is set to some large number.  This is an empirical
            # observation.

            self.queue.put(next(rollout_provider), timeout=600.0)

def construct_state():
    
    for i in range(4):
        state_ar[:,:,i]=state_q[i][:,:,0]
    return state_ar
def env_runner(env, policy, num_local_steps, summary_writer, render,log):
    """
The logic of the thread runner.  In brief, it constantly keeps on running
the policy, and as long as the rollout exceeds a certain length, the thread
runner appends the policy to the queue.
"""
    
    last_state = env.reset()    
    for _ in range(4):
        state_q.append(last_state)

    last_state = construct_state()    
    last_features = policy.get_initial_features()
    length = 0
    rewards = 0
    log_dict = dict()
    global start_rep 
    start_rep = 647907
    global count 
    count = 0
    while True:
        terminal_end = False
        rollout = PartialRollout()
        
        for _ in range(num_local_steps):
            count += 1
            #print("c",last_state.shape)
            fetched = policy.act(last_state, *last_features)
            start_rep += 1
            
            if start_rep <= constants['TRAIN_REP']:
                include = 0.0
            else:
                #print("u")
                include = 1.0
            action, value_, features = fetched[0], fetched[1], fetched[2]          

            rep = fetched[3].argmax()                         
            temp_reward = 0
            gamma = 1
            
            for _ in range(int(include)*rep + 1):
                state, reward, terminal, info = env.step(action.argmax())
                temp_reward += gamma * reward             
                gamma *=  0.99
                length += 1
                rewards += reward
                             
                if terminal:
                    break
            state_q.append(state)
            state = construct_state()

            if render:
                env.render()

            # collect the experience
            rollout.add(last_state, action, temp_reward, value_, terminal, last_features,fetched[3])           
            
            #print(features.shape)
            last_state = state
            last_features = features

            if info:
                summary = tf.Summary()
                for k, v in info.items():
                    summary.value.add(tag=k, simple_value=float(v))
                summary_writer.add_summary(summary, policy.global_step.eval())
                summary_writer.flush()

            timestep_limit = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')
            if terminal or length >= timestep_limit:
                terminal_end = True
                log_dict["episode_reward"] = rewards
                log.writekvs(log_dict)
                if length >= timestep_limit or not env.metadata.get('semantics.autoreset'):
                    last_state = env.reset()
                    for _ in range(4):
                        state_q.append(last_state)
                    last_state = construct_state()

                last_features = policy.get_initial_features()
                print("Episode finished. Sum of rewards: %d. Length: %d Steps taken: %d " % (rewards, length,start_rep))
                length = 0
                rewards = 0
		
                break
		
        if not terminal_end:
            rollout.r = policy.value(last_state, *last_features)

        # once we have enough experience, yield it, and have the ThreadRunner place it on a queue
        yield rollout

class A3C(object):
    def __init__(self, env, task, visualise,log):
        """
An implementation of the A3C algorithm that is reasonably well-tuned for the VNC environments.
Below, we will have a modest amount of complexity due to the way TensorFlow handles data parallelism.
But overall, we'll define the model, specify its inputs, and describe how the policy gradients step
should be computed.
"""
        
        self.env = env
        self.task = task
        self.lr_rate = tf.placeholder(tf.float32,[])        
        worker_device = "/job:worker/task:{}/cpu:0".format(task)
        self.lr = np.linspace(1e-3,0,100000000)
        
        with tf.device(tf.train.replica_device_setter(1, worker_device=worker_device)):
            with tf.variable_scope("global"):
                self.network = LSTMPolicy(env.observation_space.shape, env.action_space.n)
                self.global_step = tf.get_variable("global_step", [], tf.int32, initializer=tf.constant_initializer(0, dtype=tf.int32),
                                                   trainable=False)

        with tf.device(worker_device):
            with tf.variable_scope("local"):
                self.local_network = pi = LSTMPolicy(env.observation_space.shape, env.action_space.n)
                pi.global_step = self.global_step

            self.ac = tf.placeholder(tf.float32, [None, env.action_space.n], name="ac")
            self.rep = tf.placeholder(tf.float32, [None, constants['W_SIZE']], name="rep")
            self.adv = tf.placeholder(tf.float32, [None], name="adv")
            self.r = tf.placeholder(tf.float32, [None], name="r")
            self.learning_rate = tf.placeholder(tf.float32,shape = [])
            log_prob_tf = tf.nn.log_softmax(pi.logits)
            prob_tf = tf.nn.softmax(pi.logits)
            prob_tf_rep = tf.nn.softmax(pi.logits_repetition)
            log_prob_rep_tf = tf.nn.log_softmax(pi.logits_repetition)                  
            
            # the "policy gradients" loss:  its derivative is precisely the policy gradient
            # notice that self.ac is a placeholder that is provided externally.
            # adv will contain the advantages, as calculated in process_rollout
            pi_loss = - tf.reduce_sum(tf.reduce_sum(log_prob_tf * self.ac, [1]) * self.adv)
            pi_loss_rep = - tf.reduce_sum(tf.reduce_sum(log_prob_rep_tf*self.rep, [1]) * self.adv)
            # loss of value function
            vf_loss = 0.5 * tf.reduce_sum(tf.square(pi.vf - self.r))
            entropy = - tf.reduce_sum(prob_tf * log_prob_tf)
            entropy_rep = - tf.reduce_sum(prob_tf_rep * log_prob_rep_tf)

            bs = tf.to_float(tf.shape(pi.x)[0])
            self.loss = pi_loss + 0.5 * vf_loss - entropy * 0.02

            #--------defining loss function including the reptition network-------------------#
            self.loss_rep = pi_loss + 0.5 * vf_loss - entropy * 0.02 - entropy_rep * 0.02 + pi_loss_rep
            # 20 represents the number of "local steps":  the number of timesteps
            # we run the policy before we update the parameters.
            # The larger local steps is, the lower is the variance in our policy gradients estimate
            # on the one hand;  but on the other hand, we get less frequent parameter updates, which
            # slows down learning.  In this code, we found that making local steps be much
            # smaller than 20 makes the algorithm more difficult to tune and to get to work.
            self.runner = RunnerThread(env, pi, 20, visualise,log)

            var_list = [] 
            for i in pi.var_list:
                if 'repetition' not in i.name:
                    var_list.append(i)
            grads = tf.gradients(self.loss, var_list)
            self.grads_rep = tf.gradients(self.loss_rep, pi.var_list)
            if use_tf12_api:
                tf.summary.scalar("model/policy_loss", pi_loss / bs)
                tf.summary.scalar("model/value_loss", vf_loss / bs)
                tf.summary.scalar("model/entropy", entropy / bs)
                tf.summary.image("model/state", pi.x)
                tf.summary.scalar("model/grad_global_norm", tf.global_norm(grads))
                tf.summary.scalar("model/var_global_norm", tf.global_norm(pi.var_list))
                self.summary_op = tf.summary.merge_all()

            else:
                tf.scalar_summary("model/policy_loss", pi_loss / bs)
                tf.scalar_summary("model/value_loss", vf_loss / bs)
                tf.scalar_summary("model/entropy", entropy / bs)
                tf.image_summary("model/state", pi.x)
                tf.scalar_summary("model/grad_global_norm", tf.global_norm(grads))
                tf.scalar_summary("model/var_global_norm", tf.global_norm(pi.var_list))
                self.summary_op = tf.merge_all_summaries()

            grads, _ = tf.clip_by_global_norm(grads, 40.0)
            self.grads_rep, _ = tf.clip_by_global_norm(self.grads_rep, 40.0)
            # copy weights from the parameter server to the local model
            self.sync = tf.group(*[v1.assign(v2) for v1, v2 in zip(pi.var_list, self.network.var_list)])
            self.var_list_nw = []
            for i in self.network.var_list:
                 if 'repetition' not in i.name:
                     self.var_list_nw.append(i)
            grads_and_vars = list(zip(grads, self.var_list_nw))
            self.grads_and_vars_rep = list(zip(self.grads_rep, self.network.var_list))
            inc_step = self.global_step.assign_add(tf.shape(pi.x)[0])
            inc_step_rep = self.global_step.assign_add(tf.shape(pi.x)[0])           
            # each worker has a different set of adam optimizer parameters
            
            self.opt = tf.train.RMSPropOptimizer(learning_rate = self.lr_rate)
            self.train_op = tf.group(self.opt.apply_gradients(grads_and_vars), inc_step)

            #---------defining a seperate train op for the repetition network----------------#

            self.train_op_rep = tf.group(self.opt.apply_gradients(self.grads_and_vars_rep), inc_step_rep)
            self.summary_writer = None
            self.local_steps = 0

    def start(self, sess, summary_writer):
        self.runner.start_runner(sess, summary_writer)
        self.summary_writer = summary_writer

    def pull_batch_from_queue(self):
        """
self explanatory:  take a rollout from the queue of the thread runner.
"""
        rollout = self.runner.queue.get(timeout=600.0)
        while not rollout.terminal:
            try:
                rollout.extend(self.runner.queue.get_nowait())
            except queue.Empty:
                break
        return rollout

    def process(self, sess):
        """
process grabs a rollout that's been produced by the thread runner,
and updates the parameters.  The update is then sent to the parameter
server.
"""

        sess.run(self.sync)  # copy weights from shared to local
        rollout = self.pull_batch_from_queue()
        batch = process_rollout(rollout, gamma=0.99, lambda_=1.0)

        should_compute_summary = self.task == 0 and self.local_steps % 11 == 0        
        lr = self.lr[min(start_rep,99999999)]
        
        #---------------Stagewise training ----------------------#

        if start_rep <= constants['TRAIN_REP']:
            if should_compute_summary:
                fetches = [self.summary_op, self.train_op, self.global_step]
            else:
                fetches = [self.train_op, self.global_step]
        else :
           
            if should_compute_summary:
                fetches = [self.summary_op, self.train_op_rep, self.global_step]
            else:
                fetches = [self.train_op_rep, self.global_step]

       
        feed_dict = {
            self.local_network.x: batch.si,
            self.ac: batch.a,
            self.adv: batch.adv,
            self.r: batch.r,
            self.local_network.state_in[0]: batch.features[0],
            self.local_network.state_in[1]: batch.features[1],
            self.rep: batch.rep,           
            self.lr_rate:lr
        }
        
        fetched = sess.run(fetches, feed_dict=feed_dict)
        
        if should_compute_summary:
            self.summary_writer.add_summary(tf.Summary.FromString(fetched[0]), fetched[-1])
            self.summary_writer.flush()
        self.local_steps += 1
