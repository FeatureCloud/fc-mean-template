import jsonpickle
import pandas as pd
import threading
import time
import yaml
import shutil

from FeatureCloud.app.engine.app import AppState, app_state, Role
from algo import Coordinator, Client


@app_state('initial', Role.BOTH)
class InitialState(AppState):
    """
    Initialize client.
    """

    def register(self):
        self.register_transition('read input', Role.BOTH)
        
    def run(self) -> str or None:
        self.log("[CLIENT] Initializing")
        if self.id is not None:  # Test if setup has happened already
            self.log(f"[CLIENT] Coordinator {self.is_coordinator}")
            if self.is_coordinator:
                self.store('client', Coordinator())
            else:
                self.store('client', Client())
        return 'read input'
   
   
@app_state('read input', Role.BOTH)
class ReadInputState(AppState):
    """
    Read input data and config file.
    """

    def register(self):
        self.register_transition('local computation', Role.BOTH)
        
    def run(self) -> str or None:
        self.log("Read input")
        self.read_config()
        client = self.load('client')
        client.read_input(self.load('INPUT_DIR') + '/' + self.load('input_name'))
        return 'local computation'

    def read_config(self):
        self.store('INPUT_DIR', "/mnt/input")
        self.store('OUTPUT_DIR', "/mnt/output")
        with open(self.load('INPUT_DIR') + '/config.yml') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)['fc_mean']
            self.store('input_name', config['input_name'])
            self.store('output_name', config['output_name'])
        shutil.copyfile(self.load('INPUT_DIR') + "/config.yml", self.load('OUTPUT_DIR') + "/config.yml")


@app_state('local computation', Role.BOTH)
class LocalComputationState(AppState):
    """
    Compute local mean.
    """

    def register(self):
        self.register_transition('global aggregation', Role.COORDINATOR)
        self.register_transition('wait for aggregation', Role.PARTICIPANT)
        
    def run(self) -> str or None:
        self.log("Local mean computation")
        client = self.load('client')
        client.compute_local_mean()

        data_to_send = jsonpickle.encode(client.local_mean)
        self.send_data_to_coordinator(data_to_send)
        self.log(f'[CLIENT] Sending local mean data to coordinator')
        
        if self.is_coordinator:
            return 'global aggregation'
        else:
            return 'wait for aggregation'
            

@app_state('wait for aggregation', Role.PARTICIPANT)
class WaitForAggregationState(AppState):
    """
    The participant waits until it receives the global mean from the coordinator.
    """

    def register(self):
        self.register_transition('writing results', Role.PARTICIPANT)
        
    def run(self) -> str or None:
        self.log("Wait for aggregation")
        data = self.await_data()
        self.log("Received global mean from coordinator.")
        global_mean = jsonpickle.decode(data)
        client = self.load('client')
        client.set_global_mean(global_mean)
        return 'writing results'


# GLOBAL PART
@app_state('global aggregation', Role.COORDINATOR)
class GlobalAggregationState(AppState):
    """
    The coordinator receives the local mean from each client and computes the global mean.
    The coordinator broadcasts the global mean to the clients.
    """

    def register(self):
        self.register_transition('writing results', Role.COORDINATOR)
        
    def run(self) -> str or None:
        self.log("Global computation")
        data_incoming = self.gather_data()
        local_means = [jsonpickle.decode(client_data) for client_data in data_incoming]
        client = self.load('client')
        global_mean = client.compute_global_mean(local_means)
        client.set_global_mean(global_mean)
        data_to_broadcast = jsonpickle.encode(global_mean)
        self.broadcast_data(data_to_broadcast, send_to_self=False)
        self.log(f'[COORDINATOR] Broadcasting global mean to clients')
        return 'writing results'


@app_state('writing results', Role.BOTH)
class WritingResultsState(AppState):
    """
    Write results.
    """

    def register(self):
        self.register_transition('terminal', Role.PARTICIPANT)
        self.register_transition('finishing', Role.COORDINATOR)
        
    def run(self) -> str or None:
        self.log("Writing results")
        # now you can save it to a file
        self.load('client').write_results(self.load('OUTPUT_DIR') + '/' + self.load('output_name'))
        self.send_data_to_coordinator('DONE')

        if self.is_coordinator:
            return 'finishing'
        else:
            return 'terminal'


@app_state('finishing', Role.COORDINATOR)
class FinishingState(AppState):

    def register(self):
        self.register_transition('terminal', Role.COORDINATOR)
        
    def run(self) -> str or None:
        self.gather_data()
        self.log("Finishing")
        return 'terminal'
