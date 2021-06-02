import shutil
import threading
import time

import joblib
import jsonpickle
import pandas as pd
import yaml
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score, max_error, mean_absolute_error, \
    mean_absolute_percentage_error, median_absolute_error
from sklearn.model_selection import train_test_split

from app.algo import Coordinator, Client


class AppLogic:

    def __init__(self):
        # === Status of this app instance ===

        # Indicates whether there is data to share, if True make sure self.data_out is available
        self.status_available = False

        # Only relevant for coordinator, will stop execution when True
        self.status_finished = False

        # === Parameters set during setup ===
        self.id = None
        self.coordinator = None
        self.clients = None

        # === Data ===
        self.data_incoming = []
        self.data_outgoing = None

        # === Internals ===
        self.thread = None
        self.iteration = 0
        self.progress = 'not started yet'

        # === Custom ===
        self.INPUT_DIR = "/mnt/input"
        self.OUTPUT_DIR = "/mnt/output"

        self.client = None
        self.input = None
        self.sep = None
        self.label_column = None
        self.test_size = None
        self.random_state = None

    def read_config(self):
        with open(self.INPUT_DIR + '/config.yml') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)['fc_linear_regression']
            self.input = config['files']['input']
            self.sep = config['files']['sep']
            self.label_column = config['files']['label_column']
            self.test_size = config['evaluation']['test_size']
            self.random_state = config['evaluation']['random_state']

        shutil.copyfile(self.INPUT_DIR + '/config.yml', self.OUTPUT_DIR + '/config.yml')

    def handle_setup(self, client_id, coordinator, clients):
        # This method is called once upon startup and contains information about the execution context of this instance
        self.id = client_id
        self.coordinator = coordinator
        self.clients = clients
        print(f'Received setup: {self.id} {self.coordinator} {self.clients}', flush=True)

        self.thread = threading.Thread(target=self.app_flow)
        self.thread.start()

    def handle_incoming(self, data):
        # This method is called when new data arrives
        print("Process incoming data....")
        self.data_incoming.append(data.read())

    def handle_outgoing(self):
        print("Process outgoing data...")
        # This method is called when data is requested
        self.status_available = False
        return self.data_outgoing

    def app_flow(self):
        # This method contains a state machine for the client and coordinator instance

        # === States ===
        state_initializing = 1
        state_read_input = 2
        state_local_preprocessing = 3
        state_wait_for_preprocessing = 4
        state_aggregate_preprocessing = 5
        state_local_computation = 6
        state_wait_for_aggregation = 7
        state_global_aggregation = 8
        state_writing_results = 9
        state_finishing = 10

        # Initial state
        state = state_initializing
        self.progress = 'initializing...'

        while True:
            if state == state_initializing:
                print("Initializing")
                if self.id is not None:  # Test if setup has happened already
                    state = state_read_input
                    print("Coordinator", self.coordinator)
                    if self.coordinator:
                        self.client = Coordinator()
                    else:
                        self.client = Client()
            if state == state_read_input:
                print('Read input and config')
                self.read_config()

                numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
                X = pd.read_csv(self.INPUT_DIR + "/" + self.input, sep=self.sep).select_dtypes(
                    include=numerics).dropna()
                y = X.loc[:, self.label_column]
                X = X.drop(self.label_column, axis=1)

                if self.test_size is not None:
                    X, X_test, y, y_test = train_test_split(X, y, test_size=self.test_size,
                                                            random_state=self.random_state)
                    self.client.X_test = X_test
                    self.client.y_test = y_test
                self.client.X = X
                self.client.y = y

                state = state_local_preprocessing
            if state == state_local_preprocessing:
                print("Local Preprocessing")
                self.progress = 'preprocessing...'
                self.client.local_preprocessing()
                data_to_send = jsonpickle.encode(
                    [self.client.X_offset_local, self.client.y_offset_local, self.client.X_scale_local, self.client.X.shape[0]])
                if self.coordinator:
                    self.data_incoming.append(data_to_send)
                    state = state_aggregate_preprocessing
                else:
                    self.data_outgoing = data_to_send
                    self.status_available = True
                    state = state_wait_for_preprocessing
                    print(f'[CLIENT] Sending preprocessing data to coordinator', flush=True)
            if state == state_wait_for_preprocessing:
                print("Waiting for preprocessing")
                self.progress = 'wait for preprocessing'
                if len(self.data_incoming) > 0:
                    print("Received preprocess data from coordinator.")
                    data = jsonpickle.decode(self.data_incoming[0])
                    self.data_incoming = []
                    self.client.set_global_offsets(data)
                    state = state_local_computation
            if state == state_local_computation:
                print("Perform local computation")
                self.progress = 'local computation'
                xtx, xty = self.client.local_computation()

                data_to_send = jsonpickle.encode([xtx, xty])

                if self.coordinator:
                    self.data_incoming.append(data_to_send)
                    state = state_global_aggregation
                else:
                    self.data_outgoing = data_to_send
                    self.status_available = True
                    state = state_wait_for_aggregation
                    print(f'[CLIENT] Sending computation data to coordinator', flush=True)
            if state == state_wait_for_aggregation:
                print("Wait for aggregation")
                self.progress = 'wait for aggregation'
                if len(self.data_incoming) > 0:
                    print("Received aggregation data from coordinator.")
                    global_coefs = jsonpickle.decode(self.data_incoming[0])
                    self.data_incoming = []
                    self.client.set_coefs(global_coefs)
                    state = state_writing_results

            # GLOBAL PART

            if state == state_aggregate_preprocessing:
                print("Aggregate preprocessing data...")
                self.progress = 'aggregate preprocessing...'
                if len(self.data_incoming) == len(self.clients):
                    data = [jsonpickle.decode(client_data) for client_data in self.data_incoming]
                    self.data_incoming = []
                    X_offset_global, y_offset_global, X_scale_global = self.client.aggregate_preprocessing(data)
                    self.client.set_global_offsets([X_offset_global, y_offset_global, X_scale_global])
                    data_to_broadcast = jsonpickle.encode([X_offset_global, y_offset_global, X_scale_global])
                    self.data_outgoing = data_to_broadcast
                    self.status_available = True
                    state = state_local_computation
                    print(f'[COORDINATOR] Broadcasting preprocessing data to clients', flush=True)
                else:
                    print("Data from some clients still missing.")
            if state == state_global_aggregation:
                print("Global computation")
                self.progress = 'computing...'
                if len(self.data_incoming) == len(self.clients):
                    data = [jsonpickle.decode(client_data) for client_data in self.data_incoming]
                    self.data_incoming = []
                    aggregated_beta = self.client.aggregate_beta(data)
                    self.client.set_coefs(aggregated_beta)
                    data_to_broadcast = jsonpickle.encode(aggregated_beta)
                    self.data_outgoing = data_to_broadcast
                    self.status_available = True
                    state = state_writing_results
                    print(f'[COORDINATOR] Broadcasting computation data to clients', flush=True)
            if state == state_writing_results:
                print("Writing results")
                # now you can save it to a file
                print("Coef:", self.client.model.coef_)
                joblib.dump(self.client.model, self.OUTPUT_DIR + '/model.pkl')
                model = self.client.model

                if self.test_size is not None:
                    # Make predictions using the testing set
                    y_pred = model.predict(self.client.X_test)

                    # The mean squared error
                    scores = {
                        "r2_score": [r2_score(self.client.y_test, y_pred)],
                        "explained_variance_score": [explained_variance_score(self.client.y_test, y_pred)],
                        "max_error": [max_error(self.client.y_test, y_pred)],
                        "mean_absolute_error": [mean_absolute_error(self.client.y_test, y_pred)],
                        "mean_squared_error": [mean_squared_error(self.client.y_test, y_pred)],
                        "mean_absolute_percentage_error": [mean_absolute_percentage_error(self.client.y_test, y_pred)],
                        "median_absolute_error": [median_absolute_error(self.client.y_test, y_pred)]
                    }

                    scores_df = pd.DataFrame.from_dict(scores).T
                    scores_df = scores_df.rename({0: "score"}, axis=1)
                    scores_df.to_csv(self.OUTPUT_DIR + "/scores.csv")

                state = state_finishing
            if state == state_finishing:
                print("Finishing")
                self.progress = 'finishing...'
                if self.coordinator:
                    time.sleep(10)
                self.status_finished = True
                break

            time.sleep(1)


logic = AppLogic()
