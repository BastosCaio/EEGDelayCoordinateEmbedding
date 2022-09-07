# dce processing package's core functions
import scipy
import numpy as np
import matplotlib.pyplot as plt
from nolitsa import dimension, delay, utils


class DCECore(object):
    """Implements the DCE processing core functions"""

    @staticmethod
    def raw_signal_filtering(raw_data: np.ndarray, system_fs: int = 1000, output_type: str = "sos") -> np.ndarray:
        """Apply a set of filters to raw signals.

        Args:
            raw_data (nd.array): The raw EDF data.
            system_fs (int): The system frquency sampling.
            output_type (str): The butter filter result type.

        Returns:
            high_pass_filtered_data (nd.array): A filtered signal.
        """

        # defining filters
        low_pass_butter_filter = scipy.signal.butter(4, 100, "lowpass", fs=system_fs, output=output_type)
        high_pass_butter_filter = scipy.signal.butter(4, 0.5, "highpass", fs=system_fs, output=output_type)

        # applying filters (low pass filter -> high pass filter)
        low_pass_filtered_data = scipy.signal.sosfilt(low_pass_butter_filter, raw_data)
        high_pass_filtered_data = scipy.signal.sosfilt(high_pass_butter_filter, low_pass_filtered_data)

        return high_pass_filtered_data


    @staticmethod
    def calc_fnn_fraction(
        signal: np.ndarray,
        tau_value: int,
        m_dims: int,
        max_neighbors_num: int,
        frac_threshold: float = 0.10
        ) -> int:
        """Calculates the False K-Neighbors for a given signal.

        Args:
            signal (nd.array): A properly pre-processed signal.
            tau_value (int): The value for Tau.
            m_search_space_dims (list): The search space for the M-value with FNN < 10%.

        Returns:
            minimal_m (int): The estimated minimal M-value.
        """

        # defining M search space and applying fnn for f3 only (in parallel)
        m_search_space = range(1, m_dims+1)
        _, _, f3 = dimension.fnn(
            signal,
            m_search_space,
            tau_value,
            R=15.0,
            A=2.0,
            maxnum=max_neighbors_num,
            parallel=True
        )

        # returning the minimal M-value (if it is < 0.10), otherwise, returns max M-value
        min_m_idx = [idx for (idx, fraction) in enumerate(f3) if fraction < frac_threshold]

        if min_m_idx:
            return m_search_space[min_m_idx[0]]

        return m_search_space[-1]


    @staticmethod
    def calc_minimal_mutual_information(signal: np.ndarray, range_tau: int):
        """Calculates the minimal mutual information to reach the best Tau-value.

        Args:
            signal (nd.array): A properly pre-processed signal.
            range_tau (int): The range of values to test until reach the best Tau-value.

        Returns:
            mutual_info_value (int): The min value calculated for mutual information.
            mutial_info_idx (idx): The index value for the calculated min mutual information.
        """
        time_delayed_mi = delay.dmi(x=signal, maxtau=range_tau)

        min_mi_value = min(time_delayed_mi)
        min_mi_idx = np.where(time_delayed_mi == min_mi_value)

        return min_mi_value, min_mi_idx[0][0]
