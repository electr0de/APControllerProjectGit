import random
import time
import logging
from simglucose.simulation.sim_engine import SimObj
from simglucose.controller.base import Action
import numpy as np
import tensorflow as tf
import pickle
import pandas as pd
from simglucose.simulation.theta_init import ThetaInit
logger = logging.getLogger(__name__)

import keyboard


class ListForBolus:
    def __init__(self):
        self.list = []
        self.max_length = int(24*60*2 / 3)

    def append(self, item):
        if len(self.list) >= self.max_length:
            del self.list[0]
        self.list.append(item)

    def __getitem__(self, item):
        return self.list[item]

    def __setitem__(self, key, value):
        self.list[key] = value


class SimObjectForPaper(SimObj):
    def __init__(self,
                 env,
                 controller,
                 sim_time,
                 base_controller,
                 animate=True,
                 path=None,
                 previous_data=None):
        super().__init__(
                 env,
                 controller,
                 sim_time,
                 animate,
                 path)
        self.base_controller = base_controller
        self.path = "results/PaperControllerTestStuff"
        self.previous_data = previous_data
        self.plotting = False

    def save(self, stuff):
        with open(self.path+"/3dayObject.pkl", "wb") as f:
            pickle.dump(stuff, f)

    def toggle_plotting(self):
        self.plotting = not self.plotting

    def get_patient_bio(self, info):
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
        obs, reward, done, info = self.env.reset()

        keyboard.add_hotkey('p', self.toggle_plotting)

        tic = time.time()

        # bolus means IC ration
        basal_array = ListForBolus()
        bolus_array = ListForBolus()
        current_day = self.env.time.day
        food_counter = 0
        bolus_initial_list = []
        day_counter = 7
        global_state = []
        previous_food = 0

        u2ss, BW, TDI = self.get_patient_bio(info)
        theta_init = ThetaInit(u2ss, BW, TDI)

        if not self.previous_data:
            while day_counter > 0:
                if self.animate and self.plotting:
                    self.env.render()

                action = self.base_controller.policy(obs, reward, done, **info)
                obs, reward, done, info = self.env.step(action)

                basal_array.append(obs.CGM)
                bolus_array.append(obs.CGM)
                theta_init.send_glucose(obs.CGM)

                if obs.CHO != 0:
                    previous_food = obs.CHO
                if action.bolus != 0.0:
                    bolus_initial_list.append(action.bolus/previous_food)

                self.controller.current_basal_rate = action.basal

                # ignore, for debug
                state_bolus = bolus_initial_list[-1] if action.bolus != 0 else 0
                global_state.append([self.env.time, obs.CGM, obs.CHO, action.basal, action.bolus, basal_array.list[-1], bolus_array.list[-1], state_bolus])

                if current_day != self.env.time.day:
                    current_day = self.env.time.day
                    day_counter -= 1

            self.controller.theta = list(theta_init.calculate_theta())
            self.save((basal_array, bolus_array, bolus_initial_list, self.controller.current_basal_rate, self.controller.theta))
            print(f"Initialization days over, theta initialized to {self.controller.theta}")
            # input("ENTER to continue.....")
        else:
            basal_array, bolus_array, bolus_initial_list, self.controller.current_basal_rate, self.controller.theta = self.previous_data
            print("taken data from file")

        self.controller.current_breakfast_bolus = bolus_initial_list[-2-2]
        self.controller.current_lunch_bolus = bolus_initial_list[-2-1]
        self.controller.current_dinner_bolus = bolus_initial_list[-2]
        CHO_estimation_uncertainity = 0
        while self.env.time < self.env.scenario.start_time + self.sim_time:
            if self.animate and self.plotting:
                self.env.render()

            if current_day != self.env.time.day:
                basal_rate = self.controller.calculate_basal(basal_array.list[:int(24 * 60 / 3)],
                                                             basal_array.list[int(24 * 60 / 3):int(24 * 60 * 2 / 3)])
                print(f"New calculated basal is {basal_rate}")
                food_counter = 0
            else:
                basal_array.append(obs.CGM)
                basal_rate = self.controller.current_basal_rate

            if obs.CHO != 0:
                cho = random.randint(-CHO_estimation_uncertainity, CHO_estimation_uncertainity)
                temp_meal = obs.CHO + obs.CHO * cho / 100
                bolus = self.controller.calculate_bolus(bolus_array.list[:int(24*60/3)],
                                                        bolus_array.list[int(24*60/3):int(24*60*2/3)],
                                                        food_counter) * temp_meal
                print(f"New calculated bolus is {bolus}")
                food_counter += 1
            else:
                bolus_array.append(obs.CGM)
                bolus = 0.0

            current_day = self.env.time.day
            action = Action(basal=basal_rate, bolus=bolus)


            obs, reward, done, info = self.env.step(action)

        toc = time.time()
        logger.info('Simulation took {} seconds.'.format(toc - tic))

