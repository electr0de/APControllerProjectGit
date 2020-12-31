

from simglucose.analysis.report import report

from simglucose.simulation.env import T1DSimEnv
from simglucose.controller.basal_bolus_ctrller import BBController
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.simulation.sim_engine_for_paper import SimObjectForPaper
from simglucose.simulation.scenario import CustomScenario
from simglucose.simulation.sim_engine import batch_sim
from datetime import timedelta
from datetime import datetime
from simglucose.controller.PaperController import PaperRLController
import pandas as pd




path = './results/testPaperController'

sim_time = timedelta(weeks=5)

now = datetime.now()
start_time = datetime.combine(now.date(), datetime.min.time())

patient = T1DPatient.withName('adolescent#002')
sensor = CGMSensor.withName('Dexcom', seed=1)

pump = InsulinPump.withName('Insulet')
scenario=CustomScenario(start_time=start_time,sim_time=sim_time, skip_meal=False)
env = T1DSimEnv(patient, sensor, pump, scenario)

# Create a controller
RLController = PaperRLController()

basic_controller = BBController()


# Put them together to create a simulation object
s1 = SimObjectForPaper(env,RLController,sim_time,basic_controller,True,path)

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

df = pd.concat(results, keys=[s.env.patient.name for s in sim_instances])
results, ri_per_hour, zone_stats, figs, axes = report(df, path)