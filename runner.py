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
"""
class PrintStreamHandler(logging.StreamHandler):
    def __init__(self):
        logging.StreamHandler.__init__(self)

    def emit(self, record):
        msg = self.format(record)
        print(msg)



loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
handler = PrintStreamHandler()
log_format = "%(name)s - %(levelname)s - %(message)s"
handler.setFormatter(logging.Formatter(fmt=log_format))

for logger in loggers:
    logger.addHandler(handler)
"""

import pandas as pd
path = './results/testStuff'

sim_time = timedelta(weeks=1)

now = datetime.now()
start_time = datetime.combine(now.date(), datetime.min.time())

patient = T1DPatient.withName('adolescent#001')
sensor = CGMSensor.withName('Dexcom', seed=1)

pump = InsulinPump.withName('Insulet')
scenario=CustomScenario(start_time=start_time,sim_time=sim_time, skip_meal=False)
env = T1DSimEnv(patient, sensor, pump, scenario)

# Create a controller
controller = PIDController(P=0.001, I=0.00001, D=0.001, target=140)

# Put them together to create a simulation object
s1 = SimObj(env, controller, sim_time, animate=False, path=path)


patient2 = T1DPatient.withName('adolescent#002')
env2 = T1DSimEnv(patient2, sensor, pump, scenario)

s2 = SimObj(env2, controller, sim_time, animate=False, path=path)

sim_instances = [s1, s2]
results = batch_sim(sim_instances)

df = pd.concat(results, keys=[s.env.patient.name for s in sim_instances])
results, ri_per_hour, zone_stats, figs, axes = report(df, path)







