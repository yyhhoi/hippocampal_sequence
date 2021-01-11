import numpy as np

class Simulation():
    def __init__(self, t, online=True, **external_inputs):
        """

        Args:
            t (numpy.darray): with shape (time, )
            **external_inputs (numpy.darray): (t, num_neurons), entry = input values. Expected keys = 'theta', 'run'
        """
        self.t = t
        self.dt = np.abs(self.t[1] - self.t[0])
        self.num_timesteps = self.t.shape[0]
        self.current_idx = 0
        self.current_time = self.getCurrentTime()
        self.external_inputs_dict = external_inputs
        self.online = online

    def getCurrentTime(self):
        return self.t[self.current_idx]

    def get_inputs(self):
        if self.online:
            return {key:val.evaluate(self.current_idx) for key, val in self.external_inputs_dict.items()}
        else:
            return {key:val[self.current_idx] for key, val in self.external_inputs_dict.items()}
    
    def increment(self):
        self.current_idx += 1
        self.current_time = self.getCurrentTime()

    def check_end(self):
        if self.current_idx < (self.num_timesteps - 1):
            return True
        else:
            return False


