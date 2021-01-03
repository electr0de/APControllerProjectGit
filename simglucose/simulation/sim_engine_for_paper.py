import random
import time
import logging
from simglucose.simulation.sim_engine import SimObj
from simglucose.controller.base import Action
import numpy as np
import tensorflow as tf
import pickle

logger = logging.getLogger(__name__)

class ListForBolus:
    def __init__(self):
        self.list = []
        self.max_length = int(24*60*3 / 3)

    def append(self,item):
        if len(self.list)  >= self.max_length:
            del self.list[0]
            self.list.append(item)
        else:
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



    def save(self, stuff):
        with open(self.path+"/3dayObject.pkl", "wb") as f:
            pickle.dump(stuff, f)


    def simulate(self):
        obs, reward, done, info = self.env.reset()


        tic = time.time()

        #bolus means IC ration
        basal_array = ListForBolus()
        bolus_array = ListForBolus()
        basal_rate = 0.0
        bolus = 0.0
        current_day = self.env.time.day
        food_counter = 0
        bolus_initial_list = []
        day_counter = 3
        global_state = []
        previous_food = 0
        #uncomment to enable initialization from basic controller

        if not self.previous_data:
            while True:
                if current_day != self.env.time.day:
                    current_day = self.env.time.day
                    day_counter -= 1
                if day_counter == 0:
                    break

                if self.animate:
                    self.env.render()
                action = self.base_controller.policy(obs, reward, done, **info)
                obs, reward, done, info = self.env.step(action)
                basal_array.append(obs.CGM)
                bolus_array.append(obs.CGM)

                if obs.CHO != 0:
                    previous_food = obs.CHO
                if action.bolus != 0.0:
                    bolus_initial_list.append(action.bolus/previous_food)
                self.controller.current_basal_rate = action.basal

                state_bolus = bolus_initial_list[-1] if action.bolus != 0 else 0
                global_state.append([self.env.time, obs.CGM, obs.CHO, action.basal, action.bolus, basal_array.list[-1], bolus_array.list[-1], state_bolus])

            self.save((basal_array, bolus_array, bolus_initial_list, self.controller.current_basal_rate))
            input("3 days over, ENTER to continue.....")
        else:
            basal_array, bolus_array, bolus_initial_list, self.controller.current_basal_rate = self.previous_data
            print("taken data from file")

        # pickle.dump(global_state, open(self.path+"/globalstate.pkl", "wb"))


        self.controller.current_breakfast_bolus = bolus_initial_list[-2-2]
        self.controller.current_lunch_bolus = bolus_initial_list[-2-1]
        self.controller.current_dinner_bolus = bolus_initial_list[-2]
        CHO_estimation_uncertainity = 0
        while self.env.time < self.env.scenario.start_time + self.sim_time:
            if self.animate:
                self.env.render()
            if current_day == self.env.time.day:
                basal_array.append(obs.CGM)
                basal_rate = self.controller.current_basal_rate
            else:
                basal_rate = self.controller.calculate_basal(basal_array.list[:int(24*60/3)], basal_array.list[int(24*60/3):int(24*60*2/3)])
                print(f"New calculated basal is {basal_rate}")
                food_counter = 0
            if obs.CHO != 0:
                cho = random.randint(-CHO_estimation_uncertainity, CHO_estimation_uncertainity)
                temp_meal = obs.CHO + obs.CHO * cho / 100
                bolus = self.controller.calculate_bolus(bolus_array.list[:int(24*60/3)], bolus_array.list[int(24*60/3):int(24*60*2/3)], food_counter) * temp_meal
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

