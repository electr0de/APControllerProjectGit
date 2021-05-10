from gym.envs.registration import register

import matplotlib.pyplot as plt
from simglucose.analysis.report import report
from simglucose.controller.pid_ctrller import PIDController
from simglucose.simulation.env import T1DSimEnv
from simglucose.controller.basal_bolus_ctrller import BBController
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.simulation.scenario_gen import RandomScenario
from simglucose.simulation.scenario import CustomScenario,ZeroScenario
from simglucose.simulation.sim_engine import SimObj, sim, batch_sim,SimObjForKeras
from datetime import timedelta
from datetime import datetime
from examples.apply_customized_controller import MyController
import pandas as pd
register(
    id='simglucose-adolescent2-v0',
    entry_point='simglucose.envs:T1DSimEnv',
    kwargs={'patient_name': 'adolescent#002'}
)


register(
    id='simglucose-adult9-v0',
    entry_point='simglucose.envs:T1DSimEnv',
    kwargs={'patient_name': 'adult#009'}
)


register(
    id='simglucose-child5-v0',
    entry_point='simglucose.envs:T1DSimEnv',
    kwargs={'patient_name': 'child#005'}
)


path = './results/testKeras'

sim_time = timedelta(days=1)

now = datetime.now()
start_time = datetime.combine(now.date(), datetime.min.time())

patient = T1DPatient.withName('adolescent#002')
sensor = CGMSensor.withName('Dexcom', seed=1)

pump = InsulinPump.withName('Insulet')
scenario=ZeroScenario(start_time=start_time,sim_time=sim_time, skip_meal=False)
env = T1DSimEnv(patient, sensor, pump, scenario)

# Create a controller
controller1 = MyController('simglucose-adolescent2-v0')

# Put them together to create a simulation object
s1 = SimObjForKeras(env, controller1, sim_time, animate=False, path=path)

#
# patient2 = T1DPatient.withName('adult#009')
# env2 = T1DSimEnv(patient2, sensor, pump, scenario)
# controller2 = MyController('simglucose-adult9-v0')
#
# s2 = SimObj(env2, controller2, sim_time, animate=False, path=path)
#
#
# patient2 = T1DPatient.withName('child#005')
# env2 = T1DSimEnv(patient2, sensor, pump, scenario)
# controller3 = MyController('simglucose-child5-v0')
#
# s3 = SimObj(env2, controller3, sim_time, animate=False, path=path)

sim_instances = [s1]
results = batch_sim(sim_instances)
plt.plot(s1.average_reward_list)
plt.xlabel("Episode")
plt.ylabel("Avg. Epsiodic Reward")
plt.show()
plt.savefig('average_reward.png')
df = pd.concat(results, keys=[s.env.patient.name for s in sim_instances])
results, ri_per_hour, zone_stats, figs, axes = report(df, path)
