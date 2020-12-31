import time
import logging
from simglucose.simulation.sim_engine import SimObj
from simglucose.controller.base import Action
import numpy as np
import tensorflow as tf


logger = logging.getLogger(__name__)

class ListForBolus:
    def __init__(self):
        self.list = []
        self.max_length = 24*60*3

    def append(self,item):
        if len(self.list)  >= self.max_length:
            del self.list[0]
            self.list.append(item)
        else:
            self.list.append(item)

class SimObjectForPaper(SimObj):
    def __init__(self,
                 env,
                 controller,
                 sim_time,
                 base_controller,
                 animate=True,
                 path=None):
        super().__init__(
                 env,
                 controller,
                 sim_time,
                 animate,
                 path)
        self.base_controller = base_controller

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

        while True:
            if current_day != self.env.time.day:
                current_day = self.env.time.day
                day_counter -=1
            if day_counter == 0:
                break

            if self.animate:
                self.env.render()
            action = self.base_controller.policy(obs, reward, done, **info)
            obs, reward, done, info = self.env.step(action)
            basal_array.append(obs.CGM)
            bolus_array.append(obs.CGM)

            if obs.CHO != 0:
                bolus_initial_list.append(action.bolus / obs.CHO)
            self.controller.current_basal_rate = action.basal



        for i in range(3):
            if i == 0:
                self.controller.current_breakfast_bolus = bolus_initial_list[i]
            elif i == 1:
                self.controller.current_lunch_bolus = bolus_initial_list[i]
            elif i == 2:
                self.controller.current_dinner_bolus = bolus_initial_list[i]

        while self.env.time < self.env.scenario.start_time + self.sim_time:
            if self.animate:
                self.env.render()
            if current_day == self.env.time.day:
                basal_array.append(obs.CGM)
                basal_rate = self.controller.current_basal_rate
            else:
                basal_rate = self.controller.calculate_basal(basal_array.list[:24*60], basal_array.list[24*60:24*60*2])
                food_counter = 0
            if obs.CHO != 0:
                bolus = self.controller.calculate_bolus(bolus_array.list[:24*60], bolus_array.list[24*60:24*60*2], food_counter) * obs.CHO
                food_counter += 1
            else:
                bolus_array.append(obs.CGM)

            current_day = self.env.time.day
            action = Action(basal=basal_rate, bolus=bolus)
            obs, reward, done, info = self.env.step(action)

        toc = time.time()
        logger.info('Simulation took {} seconds.'.format(toc - tic))

