from simglucose.simulation.scenario import Scenario, Action

class ZeroScenario(Scenario):
    def __init__(self, start_time=None):
        super().__init__(start_time)



    def get_action(self, t):
        return Action(meal=0)

    def reset(self):
        pass