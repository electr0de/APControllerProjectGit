import logging
import time
import os
from pprint import pprint

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
            if done or obs.CGM > 180 or obs.CGM < 70:
                print(self.env.time - self.env.scenario.start_time)
                self.env.reset()
                print("dead")
                ep_reward_list.append(episodic_reward)
                episodic_reward = 0

                # Mean of last 40 episodes
                avg_reward = np.mean(ep_reward_list[-40:])
                # print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
                avg_reward_list.append(avg_reward)
                ep = ep + 1
            # print(f"Reward is {reward} insulin value is {act_in_sim_form.basal}")
            obs_in_nd = np.array([obs.CGM])
            tf_current_state = tf.expand_dims(tf.convert_to_tensor(obs_in_nd), 0)

            self.controller.learn(tf_prev_state, action, -reward, tf_current_state)

        toc = time.time()
        logger.info('Simulation took {} seconds.'.format(toc - tic))
        pprint(avg_reward_list)
        self.average_reward_list = avg_reward_list


class SimObjForKeras2(SimObj):
    def __init__(self,
                 env,
                 controller,
                 sim_time,
                 base_controller,
                 animate=False,
                 path=None,
                 sample_time=3
                 ):
        super().__init__(
            env,
            controller,
            sim_time,
            animate,
            path)
        self.sample_time = sample_time
        self.base_controller = base_controller
        self.u2ss = 0.0
        self.BW = 0.0
        self.TDI = 0.0

    def get_patient_bio(self, info):

        return self.u2ss, self.BW, self.TDI

    def set_patient_bio(self, info):
        name = info.get('patient_name')
        if any(self.base_controller.quest.Name.str.match(name)):
            q = self.base_controller.quest[self.base_controller.quest.Name.str.match(name)]
            params = self.base_controller.patient_params[self.base_controller.patient_params.Name.str.match(
                name)]
            u2ss = np.asscalar(params.u2ss.values)
            BW = np.asscalar(params.BW.values)
            TDI = q.TDI.values
        else:
            u2ss = 1.43
            BW = 57.0
            TDI = [50]

        return u2ss, BW, TDI

    def simulate(self):

        total_episode = 2000

        _, _, _, info = self.env.reset()

        self.u2ss, self.BW, self.TDI = self.set_patient_bio(info)

        tic = time.time()
        # ep = 0

        # To store reward history of each episode
        ep_reward_list = []
        # To store average reward history of last few episodes
        avg_reward_list = []

        for ep in range(total_episode):
            if self.animate:
                self.env.render()

            episodic_reward = 0

            obs, _, done, info = self.env.reset()

            glucose_rate = self.get_glucose_rate(obs.CGM, obs.CGM)

            reward = self.get_reward(obs.CGM, glucose_rate)

            IOB = 0.0

            previous_state = np.array([obs.CGM, glucose_rate, IOB])

            while True:

                tf_prev_state = tf.expand_dims(tf.convert_to_tensor(previous_state), 0)

                action = self.controller.policy(tf_prev_state, reward, done, **info)

                act_in_sim_form = Action(basal=action[0].item(0), bolus=0)

                obs, _, done, info = self.env.step(act_in_sim_form)

                glucose_rate = self.get_glucose_rate(previous_state[0], obs.CGM)

                reward = self.get_reward(obs.CGM, glucose_rate)

                IOB = self.get_IOB(info, obs.CGM, previous_state[0], previous_state[1])

                current_state = np.array([obs.CGM, glucose_rate, IOB])

                self.controller.buffer.record((previous_state, action, reward, current_state))

                episodic_reward += reward

                self.controller.buffer.learn()
                self.controller.update_actor()
                self.controller.update_critic()

                if done:

                    print("dead")
                    print(f"total time for this episode {self.env.time - self.env.scenario.start_time}")
                    self.env.reset()
                    break

                previous_state = current_state

            ep_reward_list.append(episodic_reward)

            # Mean of last 40 episodes
            avg_reward = np.mean(ep_reward_list[-40:])
            print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
            avg_reward_list.append(avg_reward)

        toc = time.time()
        logger.info('Simulation took {} seconds.'.format(toc - tic))

        plt.plot(avg_reward_list)
        plt.xlabel("Episode")
        plt.ylabel("Avg. Epsiodic Reward")
        plt.show()

    def get_reward(self, glucose, rate_of_change):

        D_l = abs(glucose - 120.0)

        # r_long = 0.0

        if 70 <= glucose <= 180:
            r_long = -D_l
        else:
            r_long = -3 * D_l

        m_target = -1 / 15

        D_r = abs(m_target * (glucose - 120) - rate_of_change)

        def get_rshort():
            if glucose < 100:
                if rate_of_change < 0.6:
                    return -5 * D_r
                elif rate_of_change >= 3:
                    return 0
                else:
                    return -3 * D_r

            if 100 <= glucose < 160:
                if rate_of_change >= 3:
                    return 0
                else:
                    return -D_r

            if 160 <= glucose < 180:
                if rate_of_change >= 3:
                    return -5 * D_r
                else:
                    return -D_r
            else:
                if rate_of_change >= 1.5:
                    return -5 * D_r
                else:
                    return -3 * D_r

        r_short = get_rshort()
        scale = 0.09

        reward = r_short + scale * r_long

        return reward

    def get_glucose_rate(self, previous_glucose, current_glucose):

        return current_glucose - previous_glucose / self.sample_time

    def get_IOB(self, info, glucose, previous_glucose, previous_rate):
        u2ss, BW, TDI = self.get_patient_bio(info)

        basal = self._calc_basal(u2ss, BW)
        u0 = self._calc_u0(basal, glucose)
        IOBbasal = self._calc_IOBbasal(u0)
        IOB_TDI = self._calc_IOB_TDI(TDI)

        dg_sig = self._calc_dg(glucose, previous_glucose)
        d2g_sig = self._calc_d2g(dg_sig, previous_rate)
        IOB_max = self._calc_IOBmax(IOBbasal, IOB_TDI, glucose, dg_sig, d2g_sig, TDI)

        return IOB_max

    def _calc_basal(self, u2ss, BW):
        return u2ss * BW / 6000 * 60

    def _calc_u0(self, basal, glucose):
        if basal >= 1.25:
            return 0.85 * basal
        if glucose >= 100:
            return 1 * basal
        if glucose < 100:
            return 0.75 * basal

    def _calc_IOBbasal(self, u0):
        aIOB = 5.0
        return aIOB * u0

    def _calc_IOB_TDI(self, TDI):
        if TDI <= 25:
            return 0.11
        if 25 < TDI <= 35:
            return 0.125
        if 35 < TDI <= 45:
            return 0.12
        if 45 < TDI <= 55:
            return 0.175
        if 55 < TDI:
            return 0.2

        raise Exception("no conditions matched")

    def _calc_dg(self, glucose, previous_glucose):
        return (glucose - previous_glucose) / self.sample_time

    def _calc_d2g(self, current_rate, previous_rate):
        return (current_rate - previous_rate) / self.sample_time

    def _calc_IOBmax(self, IOBbasal, IOB_TDI, glucose, dg_sig, d2g_sig, TDI):

        if glucose < 125:
            return 1.10 * IOBbasal

        if 150 <= glucose and dg_sig > 0.25 and d2g_sig > 0.035:
            return max(IOB_TDI, 2.5 * IOBbasal)

        if 175 <= glucose and dg_sig > 0.35 and d2g_sig > 0.035:
            return max(IOB_TDI, 3.5 * IOBbasal)

        if 200 <= glucose and dg_sig > -0.05:
            return max(IOB_TDI, 3.5 * IOBbasal)

        if 200 <= glucose and dg_sig > 0.15:
            return max(IOB_TDI, 4.5 * IOBbasal)

        if 200 <= glucose and dg_sig > 0.3:
            return max(IOB_TDI, 6.5 * IOBbasal)

        if TDI < 30:
            return 0.95 * IOBbasal

        if 125 <= glucose:
            return 1.35 * IOBbasal

        raise Exception("no conditions matched")
