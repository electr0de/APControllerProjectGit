import math
import random
from datetime import datetime, timedelta
from pprint import pprint
import numpy as np


def generate(startTime, simTime:timedelta, skip_meals:False):
    breakfast = (timedelta(hours=7), 50,True)
    lunch = (timedelta(hours=12), 60,True)
    dinner = (timedelta(hours=18,minutes=30), 80,True)
    bedtime_snack = (timedelta(hours=23), 15,False)

    meals = (breakfast,lunch,dinner,bedtime_snack)
    main_meal_variablity = 10
    snack_variability = 5

    CHO_estimation_uncertainity = 0


    main_meals_skip_per_week = 2

    meal_dict = {}

    for day in range(simTime.days):
        for meal in meals:
            time = startTime + timedelta(days=day) + meal[0]
            uncertainity = random.randint(-main_meal_variablity, main_meal_variablity) if meal[2] else random.randint(-snack_variability,snack_variability)
            cho = random.randint(-CHO_estimation_uncertainity,CHO_estimation_uncertainity)
            temp_meal = meal[1] + uncertainity
            temp_meal+=temp_meal * cho/100
            meal_dict[time] = temp_meal, meal[2]
    if skip_meals:
        for week in range(math.ceil(simTime.days/7)):
            temp = [time for time in meal_dict.keys() if startTime + timedelta(weeks=week) < time < startTime + timedelta(weeks=week+1)]
            for i in range(main_meals_skip_per_week):
                # todo this is a hack, use correct fix
                t = random.choice(temp)
                if t in meal_dict:
                    del meal_dict[t]
    
    return dict([(key,value[0]) for key, value in meal_dict.items()])

def generateDDPGTest1(startTime,simTime:timedelta, skip_meals:False ):
    meal_dict ={}
    for day in range(simTime.days):
        meal_time = np.random.normal(10, 1)
        td = timedelta(hours=meal_time)
        add_time = timedelta(hours=td.seconds//3600, minutes=(td.seconds//60)%60)
        time = startTime + timedelta(days=day) + add_time
        #currently it seems like the fed meal is 1/3 of what was intended, if that the case multiply the amount by 3
        amount = np.round(np.random.normal(65, 17)) 
        meal_dict[time] = amount

    return meal_dict




if __name__ == '__main__':
    now = datetime.now()
    start_time = datetime.combine(now.date(), datetime.min.time())
    pprint(generateDDPGTest1(start_time, timedelta(weeks=3), True))
