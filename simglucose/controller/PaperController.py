from simglucose.controller.base import Controller
from datetime import datetime,timedelta,time
import numpy as np
import math

class PaperRLController(Controller):

    def __init__(self, init_state, a_hyper = 1, a_hypo = 10, GL = 90.0, GH = 150.0):
        super().__init__(init_state)
        np.random.seed(1)

        self.a_hyper = a_hyper
        self.hypo = a_hypo
        self.GL = GL
        self.GH = GH
        self.current_basal_rate = 0.0
        self.current_breakfast_bolus = 0.0
        self.current_lunch_bolus = 0.0
        self.current_dinner_bolus = 0.0
        self.basal_theta = np.random.rand(2).tolist()
        np.random.seed(2)
        self.bolus_theta = np.random.rand(2).tolist()
        self.h = 0.5
        self.c_sigma = 0.05
        self.m = 0.5
        self.previous_basal_rate = 0.0
        np.random.seed(3)
        self.w = np.random.rand(2).tolist()
        self._lambda = 0.5
        self.gamma = 0.9
        self.z = [0.0, 0.0]
        self.a = 0.5
        self.beta = 0.5



    def extract_features(self, array):
        M_hyper = []
        M_hypo = []

        for element in array:
            if element > self.GH:
                M_hyper.append(element)
            elif element < self.GL:
                M_hypo.append(element)

        F_hyper = sum([element - self.GH for element in M_hyper]) * 1 / len(M_hyper)

        F_hypo = sum([self.GL - element for element in M_hypo]) * 1 / len(M_hypo)

        return (F_hyper, F_hypo)


    def calculate_basal(self, previous_state, basal_array):
        F_hyper, F_hypo = self.extract_features(basal_array)
        F_hyper_prev, F_hypo_prev = self.extract_features(previous_state)
        if F_hypo == 0.0:
            Ps = 0
        elif F_hypo > 0.0 and F_hyper == 0.0:
            Ps = -0.1 * F_hypo
        elif F_hypo > 0.0 and F_hyper > 0.0:
            Ps = -0.05 * F_hypo

        P = self.perform_update(Ps, (F_hyper_prev,F_hypo_prev), (F_hyper, F_hypo), True)

        self.previous_basal_rate = self.current_basal_rate


        new_basal_rate = self.current_basal_rate + self.m * P * self.current_basal_rate

        if new_basal_rate / self.current_basal_rate > 0.05:
            self.current_basal_rate += new_basal_rate*0.05
        else:
            self.current_basal_rate = new_basal_rate

    def calculate_bolus(self, previous_state, next_state, food_counter):
        F_hyper, F_hypo = self.extract_features(next_state)

        F_hyper_prev, F_hypo_prev = self.extract_features(previous_state)

        if F_hypo == 0.0:
            Ps = 0
        elif F_hypo > 0.0 and F_hyper == 0.0:
            Ps = +0.1 * F_hypo
        elif F_hypo > 0.0 and F_hyper > 0.0:
            Ps = +0.05 * F_hypo

        P = self.perform_update(Ps, (F_hyper_prev,F_hypo_prev), (F_hyper, F_hypo),False)

        if food_counter == 0:
            self.current_breakfast_bolus = self.update_bolus(self.current_breakfast_bolus, P)
            return self.current_breakfast_bolus

        elif food_counter == 1:
            self.current_lunch_bolus = self.update_bolus(self.current_lunch_bolus, P)
            return self.current_lunch_bolus

        elif food_counter == 2:
            self.current_dinner_bolus = self.update_bolus(self.current_dinner_bolus, P)
            return self.current_dinner_bolus


    def perform_update(self, Ps, F_old, F, coming_from):
        if coming_from:
            theta = self.basal_theta
        else:
            theta = self.bolus_theta

        Pa = sum([element1 * element2 for element1, element2 in zip(F, theta)])

        Pd = self.h * Pa + (1 - self.h) * Ps

        sigma = self.c_sigma * math.sqrt(F[0]**2 + F[1]**2)

        Pe = Pd + np.random.normal(0, sigma)

        cost = 1 * F[0] + 10 * F[1]
        previous_value = sum([element1 * element2 for element1, element2 in zip(F_old, self.w)])
        next_value = sum([element1 * element2 for element1, element2 in zip(F, self.w)])
        d = cost + self.gamma * next_value  - previous_value
        self.z = [self._lambda * element1 + element2 for element1,element2 in zip(self._lambda, F)]

        self.w = [element1 + self.a * d * element2 for element1,element2 in zip(self.w, self.z)]


        if coming_from:
            self.basal_theta = [element1 - self.beta * d * (Pe - Pd) / sigma**2 * self.h * element2 for element1,element2 in zip(self.basal_theta, F)]
        else:
            self.bolus_theta = [element1 - self.beta * d * (Pe - Pd) / sigma ** 2 * self.h * element2 for
                                element1, element2 in zip(self.bolus_theta, F)]
        return Pe


    def update_bolus(self, old_bolus, P):
        fusion_rate = old_bolus + self.m * P * old_bolus

        l = 1 if (self.current_basal_rate > self.previous_basal_rate and fusion_rate < old_bolus) or (self.current_basal_rate < self.previous_basal_rate and fusion_rate > old_bolus ) else 0

        new_bolus = old_bolus + (1 - l) * fusion_rate

        if new_bolus/old_bolus > 0.05:
            updated_bolus = old_bolus + new_bolus*0.05

        else:
            updated_bolus = new_bolus
        return updated_bolus



