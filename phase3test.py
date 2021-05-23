from gym.envs.registration import register

from simglucose.sensor.cgm import CGMSensor,CGMSensorWithoutNoise
from simglucose.actuator.pump import InsulinPump
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.simulation.env import T1DSimEnv
from simglucose.simulation.scenario import CustomScenario,ZeroScenario
from examples.apply_customized_controller import MyController

from simglucose.simulation.sim_engine import batch_sim,SimObjForKeras


from datetime import timedelta
from datetime import datetime


register(
    id='simglucose-adolescent2-v0',
    entry_point='simglucose.envs:T1DSimEnv',
    kwargs={'patient_name': 'adolescent#002'}
)


path = './results/phase3test1'

sim_time = timedelta(days=1)

now = datetime.now()
start_time = datetime.combine(now.date(), datetime.min.time())


patient = T1DPatient.withName('adolescent#002')
sensor = CGMSensorWithoutNoise.withName('Dexcom', seed=1)

pump = InsulinPump.withName('Insulet')

scenario = ZeroScenario(start_time=start_time,sim_time=sim_time, skip_meal=False)

env = T1DSimEnv(patient, sensor, pump, scenario)

controller1 = MyController('simglucose-adolescent2-v0')

s1 = SimObjForKeras(env, controller1, sim_time, animate=False, path=path)

sim_instances = [s1]

results = batch_sim(sim_instances)