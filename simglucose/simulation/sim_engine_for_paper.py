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
        self.max_length = 24*60*2

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
                 animate=True,
                 path=None):
        super().__init__(
                 env,
                 controller,
                 sim_time,
                 animate,
                 path)

    def simulate(self):
        obs, reward, done, info = self.env.reset()
        tic = time.time()
        basal_array = ListForBolus()
        bolus_array = ListForBolus()
        basal_rate = 0.0
        bolus = 0.0
        current_day = self.env.time.day
        food_counter = 0
        while self.env.time < self.env.scenario.start_time + self.sim_time:
            if self.animate:
                self.env.render()
            if current_day == self.env.time.day:
                basal_array.append(obs["CGM"])
                basal_rate = self.controller.current_basal_rate
            else:
                basal_rate = self.controller.calcuate_basal(basal_array.list[:24*60], basal_array.list[24*60:])
                food_counter = 0
            if obs["CHO"] != 0.0:
                bolus = self.controller.calcuate_bolus(bolus_array.list[:24*60], bolus_array.list[24*60:], food_counter)
                food_counter += 1
            else:
                bolus_array.append(obs["CGM"])

            current_day = self.env.time.day
            action = Action(basal=basal_rate, bolus=bolus)
            obs, reward, done, info = self.env.step(action)

        toc = time.time()
        logger.info('Simulation took {} seconds.'.format(toc - tic))

