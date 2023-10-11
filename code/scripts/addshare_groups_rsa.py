import os
import json
import numpy as np
import pandas as pd
import concurrent.futures
from seq_server import Server
from timeit import default_timer as timer
from tensorflow.python.keras import optimizers
from cryptography.hazmat.primitives import hashes
from tensorflow.python.keras.models import model_from_json
from cryptography.hazmat.primitives.asymmetric import padding
from code.helpers.constants import CHUNK_SIZE, NODES, ROUNDS, DATASET
from code.helpers.utils import generate_additive_shares, perform_exchange, NumpyEncoder
from code.helpers.utils import generate_groups, get_dataset_x, fetch_index, fetch_dataset
from code.helpers.utils import decode_layer, TimingCallback, get_public_key, get_private_key, NumpyDecoder

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
        self.public_key = get_public_key(self.client_id + 1)

        self.record = list()

        self.round = 0
        self.current_accuracy = 0
        self.current_training_time = 0

    def start_training(self, global_model):
        self.round += 1
        self.model = model_from_json(global_model["model_architecture"])
        self.model.compile(optimizer=optimizers.adam_v2.Adam(learning_rate=0.001), loss='categorical_crossentropy',
                           metrics=['accuracy'])
        self.model.set_weights(decode_layer(global_model["model_weights"]))

        cb = TimingCallback()

        self.model.fit(self.X_train, self.y_train, epochs=self.epochs, batch_size=10, callbacks=[cb], verbose=False,
                       use_multiprocessing=True)
        _, self.current_accuracy = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        self.current_training_time = sum(cb.logs)

    def start_sharing(self, share_id, grouping):
        model_weights = dict()

        start_time = timer()

        for layer in self.model.layers:
            if layer.trainable_weights:
                # generate additive shares
                weight_shares = list(generate_additive_shares(layer.weights[0], grouping))
                bias_shares = list(generate_additive_shares(layer.weights[1], grouping))

                # keep the i-th share equal to the client id
                self.own_share[layer.name] = [None, None]
                self.own_share[layer.name][0] = weight_shares[share_id]
                self.own_share[layer.name][1] = bias_shares[share_id]

                # replace the i-th share with None
                weight_shares[share_id] = None
                bias_shares[share_id] = None

                # assign the replaced shares to the shares to be saved in buffer for weight exchange
                model_weights[layer.name] = [None, None]
                model_weights[layer.name][0] = weight_shares
                model_weights[layer.name][1] = bias_shares

        self.secret_sharing_time = timer() - start_time

        json_str = json.dumps(model_weights, cls=NumpyEncoder)

        value_bytes = json_str.encode('utf-8')
        num_chunks = (len(value_bytes) + CHUNK_SIZE - 1) // CHUNK_SIZE

        weight_chunks = []
        for i in range(num_chunks):
            start = i * CHUNK_SIZE
            end = start + CHUNK_SIZE
            chunk = value_bytes[start:end]
            weight_chunks.append(chunk)

        encrypted_messages = []
        for json_byte_chunk in weight_chunks:
            encrypted_messages.append(
                self.public_key.encrypt(
                    json_byte_chunk,
                    padding.OAEP(
                        mgf=padding.MGF1(algorithm=hashes.SHA256()),
                        algorithm=hashes.SHA256(),
                        label=None
                    )
                )
            )

        return encrypted_messages

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

    def end_training(self, group):
        pd.DataFrame.from_dict(self.record).to_csv(
            f"resources/results/additive_groups_rsa_{group}/{self.dataset}/client_{self.client_id}.csv", index=False, header=True)


def process_end_training(client, group):
    client.end_training(group)


if __name__ == "__main__":
    print(f"EXPERIMENT: ADDSHARE GROUPS ENCRYPTED")
    GROUPS = [3, 5, 10]

    for GROUPINGS in GROUPS:
        print(f"GROUPS: {GROUPINGS}")

        for dataset in DATASET:
            print(f"DATASET: {dataset.upper()}")

            indexes = fetch_index(dataset)
            (x_train, y_train), (x_test, y_test) = fetch_dataset(dataset)

            # generate groupings
            groupings = generate_groups([i for i in range(NODES)], GROUPINGS)

            # create server
            fl_server = Server(NODES, f'additive_groups_rsa_{GROUPINGS}', dataset)

            # create clients
            clients = [AdditiveClient(i, 1, dataset, indexes[i], x_train, y_train, x_test, y_test) for i in range(NODES)]

            # start federated learning rounds
            for _ in range(ROUNDS):

                # start server
                data = fl_server.start_round()

                # start clients
                for client in clients:
                    client.start_training(data)

                # start secret sharing in groupings
                for group in groupings:
                    weights_buffer = []

                    # select client per group
                    client_buffer_group = [clients[i] for i in group]

                    # starting weight sharing
                    for i, client in enumerate(client_buffer_group):
                        node_id = group[i]
                        encrypted_messages = client.start_sharing(i, GROUPINGS)
                        private_key = get_private_key(node_id + 1)

                        decrypted_messages = []

                        for chunk in encrypted_messages:
                            decrypted_messages.append(
                                private_key.decrypt(
                                    chunk,
                                    padding.OAEP(
                                        mgf=padding.MGF1(algorithm=hashes.SHA256()),
                                        algorithm=hashes.SHA256(),
                                        label=None
                                    )
                                )
                            )

                        separator = b''
                        decoded_data = separator.join(decrypted_messages).decode('utf8')
                        weights_buffer.append(json.loads(decoded_data, cls=NumpyDecoder))

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
                            layer_weights_exchanged = perform_exchange(GROUPINGS, layer_weights)
                            layer_bias_exchanged = perform_exchange(GROUPINGS, layer_bias)

                            # reassignment each client's layer with the summed up n shares after the SMC process
                            for i in range(len(weights_buffer)):
                                weights_buffer[i][layer.name][0] = layer_weights_exchanged[i]
                                weights_buffer[i][layer.name][1] = layer_bias_exchanged[i]

                    exchange_time = timer() - start_time

                    for i, client in enumerate(client_buffer_group):
                        reassembled_weights = client.reassemble_shares(weights_buffer[i], exchange_time)
                        fl_server.fl_update(reassembled_weights)

            fl_server.end_training()

            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(process_end_training, client, GROUPINGS) for client in clients]
                concurrent.futures.wait(futures)
