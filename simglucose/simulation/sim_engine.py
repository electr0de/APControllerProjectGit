import logging
import time
import os
import tensorflow as tf
import numpy as np
from simglucose.controller.base import Action
import matplotlib.pyplot as plt
pathos = True
try:
    from pathos.multiprocessing import ProcessPool as Pool
except ImportError:
    print('You could install pathos to enable parallel simulation.')
    pathos = False

logger = logging.getLogger(__name__)


class SimObj(object):
    def __init__(self,
                 env,
                 controller,
                 sim_time,
                 animate=True,
                 path=None):
        self.env = env
        self.controller = controller
        self.sim_time = sim_time
        self.animate = animate
        self._ctrller_kwargs = None
        self.path = path

    def simulate(self):
        obs, reward, done, info = self.env.reset()
        tic = time.time()
        while self.env.time < self.env.scenario.start_time + self.sim_time:
            if self.animate:
                self.env.render()
            action = self.controller.policy(obs, reward, done, **info)
            obs, reward, done, info = self.env.step(action)
        toc = time.time()
        logger.info('Simulation took {} seconds.'.format(toc - tic))

    def results(self):
        return self.env.show_history()

    def save_results(self):
        df = self.results()
        if not os.path.isdir(self.path):
            os.makedirs(self.path)
        filename = os.path.join(self.path, str(self.env.patient.name) + '.csv')
        df.to_csv(filename)

    def reset(self):
        self.env.reset()
        self.controller.reset()


def sim(sim_object):
    print("Process ID: {}".format(os.getpid()))
    print('Simulation starts ...')
    sim_object.simulate()
    sim_object.save_results()
    print('Simulation Completed!')
    return sim_object.results()


def batch_sim(sim_instances, parallel=False):
    tic = time.time()
    if parallel and pathos:
        with Pool() as p:
            results = p.map(sim, sim_instances)
    else:
        if parallel and not pathos:
            print('Simulation is using single process even though parallel=True.')
        results = [sim(s) for s in sim_instances]
    toc = time.time()
    print('Simulation took {} sec.'.format(toc - tic))
    return results


class SimObjForKeras(SimObj):
    def __init__(self,
                 env,
                 controller,
                 sim_time,
                 animate=True,
                 path=None):
        super().__init__(
                 env,
                 controller,
                 sim_time,
                 animate,
                 path)
        self.average_reward_list = []

    def simulate(self):
        obs, reward, done, info = self.env.reset()
        tic = time.time()
        episodic_reward = 0
        ep_reward_list = []
        avg_reward_list = []
        ep = 0
        while self.env.time < self.env.scenario.start_time + self.sim_time:
            if self.animate:
                self.env.render()
            obs_in_nd = np.array([obs.CGM])
            tf_prev_state = tf.expand_dims(tf.convert_to_tensor(obs_in_nd), 0)
            previous_state = obs
            action = self.controller.policy(tf_prev_state, reward, done, **info)
            act_in_sim_form = Action(basal=action[0].item(0), bolus=0)

            obs, reward, done, info = self.env.step(act_in_sim_form)
            episodic_reward += reward
            if done:
                self.env.reset()
                print("dead")
                ep_reward_list.append(episodic_reward)
                episodic_reward = 0

                # Mean of last 40 episodes
                avg_reward = np.mean(ep_reward_list[-40:])
                #print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
                avg_reward_list.append(avg_reward)
                ep = ep + 1
            #print(f"Reward is {reward} insulin value is {act_in_sim_form.basal}")
            obs_in_nd = np.array([obs.CGM])
            tf_current_state = tf.expand_dims(tf.convert_to_tensor(obs_in_nd), 0)

            self.controller.learn(tf_prev_state, action, -reward, tf_current_state)

        toc = time.time()
        logger.info('Simulation took {} seconds.'.format(toc - tic))
        self.average_reward_list = avg_reward_list


