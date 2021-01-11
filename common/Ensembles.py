import numpy as np
from abc import ABC, abstractmethod

from .comput_utils import cos_scaled_2d


class BaseEnsemble:
    def __init__(self, simenv, num_neurons, **ensemble_params):
        self.num_neurons = num_neurons
        self.simenv = simenv
        self.ensemble_params = ensemble_params


class TsodyksEnsemble(BaseEnsemble):
    def __init__(self, simenv, num_neurons, **ensemble_params):
        """

        Args:
            simenv (Environments.Simulation):
            num_neurons (int):
            **ensemble_params:

        """
        super(TsodyksEnsemble, self).__init__(simenv, num_neurons, **ensemble_params)

        # Initialize ensemble
        self.u = np.ones(self.num_neurons) * self.ensemble_params['u_rest']
        self.u_rest = np.ones(self.num_neurons) * self.ensemble_params['u_rest']
        self.u_threshold = np.ones(self.num_neurons) * self.ensemble_params['u_threshold']
        self.tau_m = np.ones(self.num_neurons) * self.ensemble_params['tau_m']
        self.u_reset = self.ensemble_params['u_reset']

        # Initialize synaptic parameters
        self.I_ex = np.zeros(self.num_neurons)
        self.I_in = np.zeros(self.num_neurons)

        self.tau_ex = self.ensemble_params['tau_ex']  # scaler
        self.tau_in = self.ensemble_params['tau_in']  # scaler
        self.w_ex = self.ensemble_params['w_ex']  # from j to i.
        self.w_in = self.ensemble_params['w_in']  # from j to i.
        self.p_ex = self.ensemble_params['p_ex']  # scaler
        self.p_in = self.ensemble_params['p_in']  # scaler

    def state_update(self):
        fired_idx = self._synapse_dynamics()
        self._membrane_potential_dyanmics_update()
        self.simenv.increment()
        return self.simenv.check_end(), fired_idx

    def _membrane_potential_dyanmics_update(self):
        inputs_dict = self.simenv.get_inputs()
        du_dt = -(self.u - self.u_rest) \
                + self.I_ex - self.I_in \
                + inputs_dict["run"] \
                + inputs_dict["theta"]

        self.u += (du_dt / self.tau_m) * self.simenv.dt

    def _synapse_dynamics(self):
        # Threshold crossing
        fired_idx = np.where(self.u >= self.u_threshold)[0]
        self.u[fired_idx] = self.u_reset

        # Synaptic current update
        sum_ex = self._weighted_spike_sum(fired_idx, self.w_ex, self.p_ex)
        sum_in = self._weighted_spike_sum(fired_idx, self.w_in, self.p_in)
        d_Iex_dt = (-self.I_ex / self.tau_ex) + sum_ex
        d_Iin_dt = (-self.I_in / self.tau_in) + sum_in
        self.I_ex += d_Iex_dt * self.simenv.dt
        self.I_in += d_Iin_dt * self.simenv.dt
        return fired_idx

    def _weighted_spike_sum(self, fired_idx, w, p):
        fired_w = w[:, fired_idx]
        fired_success = np.random.uniform(0, 1, size=fired_w.shape) < p
        weighted_sum = np.sum(fired_w * fired_success, axis=1)
        #         fired_success = np.random.uniform(0, 1, size=self.num_neurons) < p
        #         weighted_sum = np.sum(fired_w, axis=1) * fired_success
        return weighted_sum


class RomaniEnsemble(BaseEnsemble):
    def __init__(self, simenv, num_neurons, **ensemble_params):
        super(RomaniEnsemble, self).__init__(simenv, num_neurons, **ensemble_params)
        self.m = np.ones(self.num_neurons) * self.ensemble_params['m_rest']  # Membrane potential state
        self.x = np.ones(self.num_neurons) * self.ensemble_params['x_rest']  # STD state
        self.tau = self.ensemble_params['tau']  # Membrane leak time constant
        self.tau_R = self.ensemble_params['tau_R']  # STD time constant
        self.U = self.ensemble_params['U']  # Fraction of synaptic resources
        self.alpha = self.ensemble_params['alpha']  # sigmoid constant

        self.I_R = np.zeros(self.num_neurons)  # Recurrent synaptic current
        self.w = self.ensemble_params['w']  # 2D-matrix of synaptic weight

    def state_update(self):
        self.I_R = np.mean(self.w * self.m.reshape(1, -1) * self.x.reshape(1, -1),
                           axis=1)

        inputs_dict = self.simenv.get_inputs()
        total_I = self.I_R + inputs_dict['I_E']
        f = total_I
        f[total_I < 5] = self.alpha * np.log(1 + np.exp(total_I[total_I<5] / self.alpha))

        dm_dt = (-self.m + f) / self.tau

        dx_dt = (1 - self.x) / self.tau_R - (self.U * self.x * self.m)

        self.m += dm_dt * self.simenv.dt
        self.x += dx_dt * self.simenv.dt
        self.simenv.increment()

        # Firing, by poisson process
        rand_num = np.random.uniform(size=self.num_neurons)
        firing_idx = np.where(rand_num < (self.m * self.simenv.dt))[0]

        return self.simenv.check_end(), self.m.copy(), firing_idx


class BaseInputEvaluator(ABC):
    def __init__(self, pos, neuron_pos, t, f_theta, I_theta, I, I_L):
        """

        Parameters
        ----------
        pos : ndarray
            x, y positions of the rat. Shape = (time, 2)
        neuron_pos : ndarray
            x, y positions of the neurons. Shape = (num_neurons, 2)
        t : ndarray
            Times of the positions.
        f_theta : scalar
            Frequency of theta modulation.
        I_theta : scalar
            Amplitude of theta modulation.
        I : scalar
            Amplitude of base input current.
        I_L : scalar
            Amplitude of place-specific current
        """
        self.pos = pos
        self.neuron_pos = neuron_pos
        self.t = t
        self.num_neurons = neuron_pos.shape[0]
        self.f_theta = f_theta
        self.I_theta = I_theta
        self.I = I
        self.I_L = I_L
        super().__init__()

    @abstractmethod
    def evaluate(self, tidx):
        """

        Parameters
        ----------
        tidx : int
            Index of the time of the input

        Returns
        -------
        I_E : ndarray
            1D array of external input current with shape (num_neurons, )
        """
        return None


class InputEvaluatorCustom(BaseInputEvaluator):
    def __init__(self, pos, neuron_pos, t, gt_func, f_theta, I_theta, I, I_L):
        super(InputEvaluatorCustom, self).__init__(pos, neuron_pos, t, f_theta, I_theta, I, I_L)
        self.gt_func = gt_func

    def evaluate(self, tidx):
        gt = self.gt_func(self.pos[tidx, ], self.neuron_pos)
        theta_input = self.I_theta * np.cos(2 * np.pi * self.f_theta * self.t[tidx])  # scalar
        I_E = np.ones(self.num_neurons) * self.I + theta_input + self.I_L * gt  # (num_neurons)
        return I_E


class InputEvaluatorToroidal(BaseInputEvaluator):
    def __init__(self, pos, neuron_pos, t, f_theta, I_theta, I, I_L):
        super(InputEvaluatorToroidal, self).__init__(pos, neuron_pos, t, f_theta, I_theta, I, I_L)

    def evaluate(self, tidx):
        gt = np.cos(self.pos[tidx, 0] - self.neuron_pos[:, 0]) + np.cos(
            self.pos[tidx, 1] - self.neuron_pos[:, 1])  # (num_neurons)
        theta_input = self.I_theta * np.cos(2 * np.pi * self.f_theta * self.t[tidx])  # scalar
        I_E = np.ones(self.num_neurons) * self.I + theta_input + self.I_L * gt  # (num_neurons)
        return I_E


class InputEvaluatorSquare(BaseInputEvaluator):
    def __init__(self, pos, neuron_pos, t, f_theta, I_theta, I, I_L, lambda_E):
        super(InputEvaluatorSquare, self).__init__(pos, neuron_pos, t, f_theta, I_theta, I, I_L)
        self.lambda_E = lambda_E

    def evaluate(self, tidx):
        exponent = np.sqrt(np.sum(np.square(self.pos[tidx,].reshape(1, 2) - self.neuron_pos), axis=1))
        gt = np.exp(-exponent / self.lambda_E)  # (num_neurons, )
        theta_input = self.I_theta * np.cos(2 * np.pi * self.f_theta * self.t[tidx])  # scalar
        I_E = np.ones(self.num_neurons) * self.I + theta_input + self.I_L * gt  # (num_neurons)
        return I_E



