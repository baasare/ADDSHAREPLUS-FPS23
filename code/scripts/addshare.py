import os
import numpy as np
import pandas as pd
import concurrent.futures
from seq_server import Server
from timeit import default_timer as timer
from tensorflow import keras
import sys

sys.path.append('../helpers')

from constants import NODES, ROUNDS, EPOCHS, DATASET
from utils import generate_additive_shares, perform_exchange
from utils import decode_layer, TimingCallback, get_dataset_x, fetch_dataset, fetch_index

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class AdditiveClient:
    def __init__(self, client_id, epochs, dataset, index, x_train, y_train, x_test, y_test):

        self.client_id = client_id
        self.dataset = dataset
        self.server = None

        self.X_train, self.y_train, self.X_test, self.y_test = \
            get_dataset_x(index, dataset, x_train, y_train, x_test, y_test)
        self.model = None
        self.epochs = epochs

        self.own_share = dict()
        self.secret_sharing_time = None

        self.record = list()

        self.round = 0
        self.current_accuracy = 0
        self.current_training_time = 0

    def start_training(self, global_model):
        self.round += 1
        self.model = keras.models.model_from_json(global_model["model_architecture"])
        self.model.compile(optimizer=keras.optimizers.legacy.Adam(learning_rate=0.001), loss='categorical_crossentropy',
                           metrics=['accuracy'])
        self.model.set_weights(decode_layer(global_model["model_weights"]))

        cb = TimingCallback()

        self.model.fit(self.X_train, self.y_train, epochs=self.epochs, batch_size=10, callbacks=[cb], verbose=False,
                       use_multiprocessing=True)
        _, self.current_accuracy = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        self.current_training_time = sum(cb.logs)

        model_weights = dict()

        start_time = timer()

        for layer in self.model.layers:
            if layer.trainable_weights:
                # generate additive shares
                weight_shares = list(generate_additive_shares(layer.weights[0], NODES))
                bias_shares = list(generate_additive_shares(layer.weights[1], NODES))

                # keep the i-th share equal to the client id
                self.own_share[layer.name] = [None, None]
                self.own_share[layer.name][0] = weight_shares[self.client_id]
                self.own_share[layer.name][1] = bias_shares[self.client_id]

                # replace the i-th share with None
                weight_shares[self.client_id] = None
                bias_shares[self.client_id] = None

                # assign the replaced shares to the shares to be saved in buffer for weight exchange
                model_weights[layer.name] = [None, None]
                model_weights[layer.name][0] = weight_shares
                model_weights[layer.name][1] = bias_shares

        self.secret_sharing_time = timer() - start_time
        return model_weights

    def reassemble_shares(self, data, exchange_time):
        start_time = timer()

        model_weights = dict()
        layer_weights = dict()
        for layer in data.keys():
            weight_bias = data[layer]
            weight_bias[0][0] = self.own_share[layer][0]
            weight_bias[1][0] = self.own_share[layer][1]

            model_weights[layer] = [None, None]
            model_weights[layer][0] = weight_bias[0]
            model_weights[layer][1] = weight_bias[1]

            temp_weight_bias = [None, None]
            temp_weight_bias[0] = np.sum((model_weights[layer][0]), axis=0)
            temp_weight_bias[1] = np.sum((model_weights[layer][1]), axis=0)

            layer_weights[layer] = temp_weight_bias
        self.secret_sharing_time = self.secret_sharing_time + exchange_time + (timer() - start_time)

        self.record.append({
            'round': self.round,
            'accuracy': self.current_accuracy,
            'training': self.current_training_time,
            'secret_sharing': self.secret_sharing_time
        })

        return layer_weights

    def end_training(self):
        pd.DataFrame.from_dict(self.record).to_csv(
            f"../../results/numerical/additive/{self.dataset}/client_{self.client_id}.csv",
            index=False, header=True)


def process_end_training(client):
    client.end_training()


if __name__ == "__main__":
    print(f"EXPERIMENT: ADDSHARE")
    for dataset in DATASET:
        print(f"DATASET: {dataset.upper()}")
        indexes = fetch_index(dataset)
        (x_train, y_train), (x_test, y_test) = fetch_dataset(dataset)

        fl_server = Server(NODES, 'additive', dataset)

        clients = [AdditiveClient(i, EPOCHS, dataset, indexes[i], x_train, y_train, x_test, y_test) for i in
                   range(NODES)]

        for _ in range(ROUNDS):
            weights_buffer = []
            fl_update = fl_server.start_round()
            for client in clients:
                weights_buffer.append(client.start_training(fl_update))

            start_time = timer()

            for layer in fl_server.global_model.layers:
                if layer.trainable_weights:
                    layer_weights, layer_bias = [], []
                    layer_weights_exchanged, layer_bias_exchanged = [], []

                    # for this layer get each clients layer weights and bias shares
                    for client in weights_buffer:
                        layer_weights.append(client[layer.name][0])
                        layer_bias.append(client[layer.name][1])

                    # perform exchange of secret sharing and exchange.
                    layer_weights_exchanged = perform_exchange(NODES, layer_weights)
                    layer_bias_exchanged = perform_exchange(NODES, layer_bias)

                    # reassignment each client's layer with the summed up n shares after the SMC process
                    for i in range(len(weights_buffer)):
                        weights_buffer[i][layer.name][0] = layer_weights_exchanged[i]
                        weights_buffer[i][layer.name][1] = layer_bias_exchanged[i]

            exchange_time = timer() - start_time

            for i, client in enumerate(clients):
                reassembled_weights = client.reassemble_shares(weights_buffer[i], exchange_time)
                fl_server.fl_update(reassembled_weights)

        fl_server.end_training()

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_end_training, client) for client in clients]
            concurrent.futures.wait(futures)
