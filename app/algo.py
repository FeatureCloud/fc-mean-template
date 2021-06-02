import numpy as np
import pandas as pd

INPUT_PATH = "/mnt/input/data.csv"
OUTPUT_PATH = "/mnt/output/result.txt"


class Client:
    input_data = None
    number_of_samples = None
    local_sum = None
    local_mean = None
    global_mean = None

    def __init__(self):
        pass

    def read_input(self):
        try:
            self.input_data = pd.read_csv(INPUT_PATH, header=None)
        except FileNotFoundError:
            print(f'File {INPUT_PATH} could not be found.', flush=True)
            exit()
        except Exception as e:
            print(f'File could not be parsed: {e}', flush=True)
            exit()

    def compute_local_mean(self):
        self.number_of_samples = self.input_data.shape[1]
        self.local_sum = self.input_data.to_numpy().sum()
        print(f'Local sum: {self.local_sum}', flush=True)
        self.local_mean = self.local_sum / self.number_of_samples
        print(f'Local mean: {self.local_mean}', flush=True)

    def set_global_mean(self, global_mean):
        self.global_mean = global_mean

    def write_results(self):
        f = open(OUTPUT_PATH, "a")
        f.write(str(self.global_mean))
        f.close()


class Coordinator(Client):

    def compute_global_mean(self, local_means):
        return np.sum(local_means) / len(local_means)
