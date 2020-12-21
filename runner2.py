from gym.envs.registration import register


from simglucose.analysis.report import report
from simglucose.controller.pid_ctrller import PIDController
from simglucose.simulation.env import T1DSimEnv
from simglucose.controller.basal_bolus_ctrller import BBController
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.simulation.scenario_gen import RandomScenario
from simglucose.simulation.scenario import CustomScenario
from simglucose.simulation.sim_engine import SimObj, sim, batch_sim
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

sim_time = timedelta(days=3)

now = datetime.now()
start_time = datetime.combine(now.date(), datetime.min.time())

patient = T1DPatient.withName('adolescent#002')
sensor = CGMSensor.withName('Dexcom', seed=1)

pump = InsulinPump.withName('Insulet')
scenario=CustomScenario(start_time=start_time,sim_time=sim_time, skip_meal=False)
env = T1DSimEnv(patient, sensor, pump, scenario)

# Create a controller
controller1 = MyController('simglucose-adolescent2-v0')

# Put them together to create a simulation object
s1 = SimObj(env, controller1, sim_time, animate=False, path=path)


patient2 = T1DPatient.withName('adult#009')
env2 = T1DSimEnv(patient2, sensor, pump, scenario)
controller2 = MyController('simglucose-adult9-v0')

s2 = SimObj(env2, controller2, sim_time, animate=False, path=path)


patient2 = T1DPatient.withName('child#005')
env2 = T1DSimEnv(patient2, sensor, pump, scenario)
controller3 = MyController('simglucose-child5-v0')

s3 = SimObj(env2, controller3, sim_time, animate=False, path=path)

sim_instances = [s1, s2, s3]
results = batch_sim(sim_instances)

df = pd.concat(results, keys=[s.env.patient.name for s in sim_instances])
results, ri_per_hour, zone_stats, figs, axes = report(df, path)
