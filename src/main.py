# dce processing package's main file
import os
import re
import time

import mne
import h5py
from nolitsa import utils

from core import DCECore


class Application(object):
    """
    Implements a Mini CLI to load the data to be processed.
    """
    DEFAULT_TAU_VALUE = 25
    DEFAULT_TAU_RANGE = 90
    DEFAULT_M_DIMS = 9
    DEFAULT_MAX_NEIGHBORS_NUM = 700

    def __init__(self):
        """
        Constructor method for loading the EDF file,
        as well as calling the application's core functions.
        """

        # asking for user input data
        self.file_path = input("Paste the EDF file path: ")
        self.file_output_path = input("Paste the path to save your file into: ")

        # extracting raw & metadata and generating a name for the output file
        try:
            signals, channels, n_electrodes, info = self.__load_raw_data()
            output_file_name = self.__generate_output_file_name()
        except:
            raise("The file you're trying to load doesn't exist!")

        # iterating over each electrode
        print("Starting DCE Processing...")
        start_proc_time = time.time()

        for electrode in range(n_electrodes+1):
            # reading electrode data
            filtered_data = signals[electrode, :]

            # estimating signal minimal M-value
            minimal_m = DCECore.calc_fnn_fraction(
                filtered_data,
                self.DEFAULT_TAU_VALUE,
                self.DEFAULT_M_DIMS,
                self.DEFAULT_MAX_NEIGHBORS_NUM
            )
            print(f"Minimal M-value for Electrode {electrode} is: {minimal_m}")

            # calculating mutual info for the best Tau-value
            _, min_mi_idx = DCECore.calc_minimal_mutual_information(filtered_data, self.DEFAULT_TAU_RANGE)
            print(f"Mutual information for electrode {electrode} just got calculated!")

            # calc delayed array for filtered data based on minimal_m and min_mi_idx (DCE)
            print(f"Calculating DCE for electrode {electrode}")
            delayed_array = utils.reconstruct(filtered_data, minimal_m, min_mi_idx)

            # serializing the processed data and metadata info in a h5py object
            output_file = h5py.File(os.path.join(self.file_output_path, output_file_name), "w")
            output_file.create_dataset(f"electrode_{electrode}: {channels[electrode-1]}", data=delayed_array)
            output_file.attrs["minimal_m"] = minimal_m
            output_file.attrs["min_mutial_info_idx"] = min_mi_idx
            output_file.close()

        end_proc_time = time.time()
        print("DCE Processing finished! The new data was saved successfully!")
        print(f"It took {end_proc_time - start_proc_time}s to complete!")



    def __load_raw_data(self):
        """Loads & extracts raw data file and it's metadata"""
        raw_data = mne.io.read_raw_edf(self.file_path)

        signals = raw_data.get_data()
        channels = raw_data.ch_names
        n_electrodes = len(channels)
        info = raw_data.info

        return signals, channels, n_electrodes, info


    def __generate_output_file_name(self):
        """Generates a name for the output file."""
        pattern = re.split("/", self.file_path)

        return pattern[len(pattern)-1][:-4]


if __name__ == "__main__":
    app = Application()
