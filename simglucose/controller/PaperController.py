from functools import partial
from pprint import pprint
import matplotlib.pyplot as plt

#import test2
from simglucose.controller.base import Controller
from datetime import datetime, timedelta, time
import numpy as np
import math

percent_value = 0.05

sign = lambda x: math.copysign(1, x)

normalize_f = lambda x: (x - 39) / (600 - 39)

class PaperRLController(Controller):

    def __init__(self, a_hyper=1, a_hypo=10, current_breakfast_bolus=0.0, current_lunch_bolus=0.0,
                 current_dinner_bolus=0.0, current_basal_rate=0.0,current_snack_bolus = 0.0,  init_state=None):
        super().__init__(init_state)
        np.random.seed(1)

        self.a_hyper = a_hyper
        self.hypo = a_hypo
        self.GL = normalize_f(90)
        self.GH = normalize_f(150)
        self.current_basal_rate = current_basal_rate
        self.current_breakfast_bolus = current_breakfast_bolus  # bolus means IC ratio
        self.current_lunch_bolus = current_lunch_bolus
        self.current_dinner_bolus = current_dinner_bolus
        #self.current_snack_bolus = current_snack_bolus
        self.basal_theta = []
        self.bolus_theta = []
        #np.random.seed(2)
        #self.bolus_theta = np.random.rand(2).tolist()
        self.h = 0.5
        self.c_sigma = 0.05
        self.m = 0.5
        self.previous_basal_rate = 0.0
        np.random.seed(55)
        self.w = (np.random.rand(2)*2-1).tolist()
        self._lambda = 0.5
        self.gamma = 0.9
        self.z = [0.0, 0.0]
        self.a = 0.5
        self.beta = 0.5
        self.beta_basal = 0.5
        self.value_factor = 10
        # self.time_array = []
        # self.theta_array_1 = []
        # self.theta_array_2 = []
        # self.bolus_time_array = []
        # self.F_1_array = []
        # self.F_2_array = []
        #plt.figure(200)
        #self.fig, self.axis = plt.subplots(4)
        #plt.show()
        #self.axis[0].set_title(" Hyper feature for basal")
        #self.axis[1].set_title(" Hypo feature for basal")
        #self.axis[2].set_title("Hyper theta for basal")
        #self.axis[3].set_title(" Hypo theta for basal")

    def extract_features(self, array):
        M_hyper = []
        M_hypo = []

        for element in array:
            if element > 150:
                M_hyper.append(normalize_f(element))
            elif element < 90:
                M_hypo.append(normalize_f(element))

        F_hyper = sum([element - self.GH for element in M_hyper]) * 1 / len(M_hyper) if M_hyper else 0

        F_hypo = sum([self.GL - element for element in M_hypo]) * 1 / len(M_hypo) if M_hypo else 0

        return (F_hyper, F_hypo)

    def calculate_basal(self, previous_state, basal_array, time):
        F_hyper, F_hypo = self.extract_features(basal_array)
        F_hyper_prev, F_hypo_prev = self.extract_features(previous_state)

        #
        # self.F_1_array.append(F_hyper)
        # self.F_2_array.append(F_hypo)
        # self.time_array.append(time)
        #
        # self.axis[0].plot(self.time_array, self.F_1_array)
        #
        # self.axis[1].plot(self.time_array, self.F_2_array)
        #
        # plt.pause(0.001)

        Ps = None
        if F_hypo == 0.0:
            Ps = 0
        elif F_hypo > 0.0 and F_hyper == 0.0:
            Ps = -0.1 * F_hypo
        elif F_hypo > 0.0 and F_hyper > 0.0:
            Ps = -0.05 * F_hypo

        assert Ps is not None, "No conditions matched"

        P = self.perform_update(Ps, (F_hyper_prev, F_hypo_prev), (F_hyper, F_hypo), True)

        self.previous_basal_rate = self.current_basal_rate

        br_change = self.m * P * self.current_basal_rate

        # uncomment to enable 5 % change
        #percent_value = 0
        if abs(br_change / self.current_basal_rate) > percent_value:
            self.current_basal_rate += self.current_basal_rate * percent_value * sign(br_change)
            print(" used % changed")
        else:
            self.current_basal_rate += br_change
            print(" didn't use % changed")
        return self.current_basal_rate

    def calculate_bolus(self, previous_state, next_state, food_counter, time):
        F_hyper, F_hypo = self.extract_features(next_state)

        F_hyper_prev, F_hypo_prev = self.extract_features(previous_state)

        #
        # self.F_1_array.append(F_hyper)
        # self.F_2_array.append(F_hypo)
        # self.bolus_time_array.append(time)
        #
        # self.axis[0].plot(self.bolus_time_array, self.F_1_array)
        # self.axis[1].plot(self.bolus_time_array, self.F_2_array)


        Ps = None
        if F_hypo == 0.0:
            Ps = 0
        elif F_hypo > 0.0 and F_hyper == 0.0:
            Ps = +0.1 * F_hypo
        elif F_hypo > 0.0 and F_hyper > 0.0:
            Ps = +0.05 * F_hypo

        assert Ps is not None, "No conditions matched"

        P = self.perform_update(Ps, (F_hyper_prev, F_hypo_prev), (F_hyper, F_hypo), False)

        if food_counter == 0:
            self.current_breakfast_bolus = self.update_bolus(self.current_breakfast_bolus, P)
            return self.current_breakfast_bolus

        if food_counter == 1:
            self.current_lunch_bolus = self.update_bolus(self.current_lunch_bolus, P)
            return self.current_lunch_bolus

        if food_counter == 2:
            self.current_dinner_bolus = self.update_bolus(self.current_dinner_bolus, P)
            return self.current_dinner_bolus
        #if food_counter == 3:
            #self.current_snack_bolus = self.update_bolus(self.current_snack_bolus, P)
            #return self.current_snack_bolus
        return 0.0

    def perform_update(self, Ps, F_old, F, coming_from):

        if coming_from:
            theta = self.basal_theta
        else:
            theta = self.bolus_theta

        #theta = self.theta

        print(f"theta: {theta}")

        Pa = sum([element1 * element2 for element1, element2 in zip(F, theta)])

        Pd = self.h * Pa + (1 - self.h) * Ps

        sigma = self.c_sigma * (F[0] ** 2 + F[1] ** 2)

        Pe = Pd + np.random.normal(0, sigma)

        cost = 1 * F[0] + self.value_factor * F[1]
        previous_value = sum([element1 * element2 for element1, element2 in zip(F_old, self.w)])
        next_value = sum([element1 * element2 for element1, element2 in zip(F, self.w)])
        d = cost + self.gamma * next_value - previous_value

        self.w = [element1 + self.a * d * element2 for element1, element2 in zip(self.w, self.z)]

        self.z = [self._lambda * element1 + element2 for element1, element2 in zip(self.z, F)]

        if coming_from:
            self.basal_theta = [element1 - self.beta_basal * d * (Pe - Pd) / sigma ** 2 * self.h * element2 for
                                element1, element2 in zip(self.basal_theta, F)]
        else:
            self.bolus_theta = [element1 - self.beta * d * (Pe - Pd) / sigma ** 2 * self.h * element2 for
                                element1, element2 in zip(self.bolus_theta, F)]

        assert sigma > 0.0000001, "sigma is too low"
        # self.theta_array_1.append(self.theta[0])
        # self.theta_array_2.append(self.theta[1])
        # self.axis[2].plot(self.time_array, self.theta_array_1)
        # self.axis[3].plot(self.time_array, self.theta_array_2)



        return Pe

    def update_bolus(self, old_bolus, P):
        fusion_rate = old_bolus + self.m * P * old_bolus

        l = 1 if (self.current_basal_rate > self.previous_basal_rate and fusion_rate < old_bolus) or (
                    self.current_basal_rate < self.previous_basal_rate and fusion_rate > old_bolus) else 0
        
        bl_change = (1 - l) * fusion_rate

        if abs(bl_change / old_bolus) > percent_value:
            old_bolus += sign(bl_change) * old_bolus * percent_value
            print(" used % changed")
        else:
            old_bolus += bl_change
            print(" didn't use % changed")
        return old_bolus




# if __name__ == '__main__':
#
#     GL = normalize_f(90)
#     GH = normalize_f(150)
#
#     def extract_features(array):
#         M_hyper = []
#         M_hypo = []
#
#         for element in array:
#             if element > 150:
#                 M_hyper.append(normalize_f(element))
#             elif element < 90:
#                 M_hypo.append(normalize_f(element))
#
#         F_hyper = sum([element - GH for element in M_hyper]) * 1 / len(M_hyper) if M_hyper else 0
#
#         F_hypo = sum([GL - element for element in M_hypo]) * 1 / len(M_hypo) if M_hypo else 0
#
#         return (F_hyper, F_hypo)
#
#     array = test2.array
#     print(extract_features(array))