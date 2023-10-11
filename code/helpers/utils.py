import os
import json
import codecs
import pickle
import random
import logging
import itertools
import numpy as np
import pandas as pd
from PIL import Image
import scipy.io as sio
import tensorflow as tf
from decimal import Decimal
from keras.callbacks import Callback
from keras.utils import to_categorical
from timeit import default_timer as timer
from keras.datasets import cifar10, mnist, fashion_mnist
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from constants import FIELD_SIZE, BIT_SIZE, DATASET


def decode_layer(b64_str):
    return pickle.loads(codecs.decode(b64_str.encode(), "base64"))


def encode_layer(layer):
    return codecs.encode(pickle.dumps(layer), "base64").decode()


def generate_keys(save_path, name, nbits=BIT_SIZE):
    """
    :param save_path:
    :param nbits:
    :param name:
    :return:
    """
    os.makedirs(save_path, exist_ok=True)

    # Generate an RSA key pair
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=nbits
    )
    public_key = private_key.public_key()

    public_pem_path = os.path.join(save_path, 'client_' + str(name) + '_public.pem')
    private_pem_path = os.path.join(save_path, 'client_' + str(name) + '_private.pem')

    try:
        # Serialize the private key to PEM format and save it to a file
        with open(private_pem_path, 'wb') as f:
            f.write(private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            ))

        # Serialize the public key to PEM format and save it to a file
        with open(public_pem_path, 'wb') as f:
            f.write(public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            ))
    except Exception as ex:
        logging.error(ex)

    return public_pem_path, private_pem_path


def get_public_key(client_id):
    current_dir = os.path.dirname(os.getcwd())
    path = current_dir + f'/resources/keys/client_{str(client_id)}_public.pem'

    with open(path, 'rb') as f:
        return serialization.load_pem_public_key(f.read())


def get_private_key(client_id):
    current_dir = os.path.dirname(os.getcwd())
    path = current_dir + f'/resources/keys/client_{str(client_id)}_private.pem'

    with open(path, 'rb') as f:
        return serialization.load_pem_private_key(f.read(), password=None)


def get_dataset(client_id, dataset="cifar-10", iid="iid", balanced='balanced'):
    current_dir = os.path.dirname(os.getcwd())
    path = current_dir + f'/resources/dataset/{dataset}/{iid}_{balanced}.txt'
    clients_indexes = np.loadtxt(path, dtype=object)
    clients_indexes = clients_indexes.astype(np.float64)
    clients_indexes = clients_indexes.astype(np.int64)

    if client_id == 'server':
        client_id = 0

    if dataset == "cifar-10":
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train = x_train[clients_indexes[client_id]]
        y_train = y_train[clients_indexes[client_id]]

        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0

        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)

    elif dataset == 'svhn':
        train_data = sio.loadmat(current_dir + f'/resources/dataset/svhn/train_32x32.mat')
        test_data = sio.loadmat(current_dir + f'/resources/dataset/svhn/test_32x32.mat')

        x_train = np.array(train_data['X'])
        y_train = np.array(train_data['y'])

        x_test = np.array(test_data['X'])
        y_test = np.array(test_data['y'])

        x_train = np.moveaxis(x_train, -1, 0)
        x_test = np.moveaxis(x_test, -1, 0)

        x_train = x_train[clients_indexes[client_id]]
        y_train = y_train[clients_indexes[client_id]]

        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0

        y_train = to_categorical((y_train - 1), 10)
        y_test = to_categorical((y_test - 1), 10)

    else:
        (x_train, y_train), (x_test, y_test) = mnist.load_data() if dataset == "mnist" else fashion_mnist.load_data()

        x_train = x_train[clients_indexes[client_id]]
        y_train = y_train[clients_indexes[client_id]]

        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255.0
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255.0

        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)

    return x_train, y_train, x_test, y_test


def get_dataset_x(index, dataset, x_train, y_train, x_test, y_test):
    if dataset == "cifar-10":
        x_train = x_train[index]
        y_train = y_train[index]

        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0

        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)

    elif dataset == 'svhn':
        x_train = np.moveaxis(x_train, -1, 0)
        x_test = np.moveaxis(x_test, -1, 0)

        x_train = x_train[index]
        y_train = y_train[index]

        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0

        y_train = to_categorical((y_train - 1), 10)
        y_test = to_categorical((y_test - 1), 10)

    else:
        x_train = x_train[index]
        y_train = y_train[index]

        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255.0
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255.0

        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)

    return x_train, y_train, x_test, y_test


def fetch_dataset(dataset):
    if dataset == "cifar-10":
        return cifar10.load_data()
    elif dataset == "mnist":
        return mnist.load_data()
    elif dataset == "f-mnist":
        return fashion_mnist.load_data()
    else:
        current_dir = os.path.dirname(os.getcwd())
        train_data = sio.loadmat(current_dir + f'/resources/dataset/svhn/train_32x32.mat')
        test_data = sio.loadmat(current_dir + f'/resources/dataset/svhn/test_32x32.mat')
        x_train = np.array(train_data['X'])
        y_train = np.array(train_data['y'])

        x_test = np.array(test_data['X'])
        y_test = np.array(test_data['y'])

        return (x_train, y_train), (x_test, y_test)


def fetch_index(dataset):
    current_dir = os.path.dirname(os.getcwd())
    path = current_dir + f'/resources/dataset/{dataset}/iid_balanced.txt'
    clients = np.loadtxt(path, dtype=object)
    clients = clients.astype(np.float64)
    clients = clients.astype(np.int64)
    return clients


def iid_balanced(client_number, train_size, dataset):
    rand_array = np.arange(train_size)
    np.random.shuffle(rand_array)

    clients = [[] for _ in range(client_number)]

    for i in range(client_number):
        clients[i] = rand_array[
                     int(i * train_size / client_number):int((i + 1) * train_size / client_number)]

    np.savetxt(f"../resources/dataset/{dataset}/iid_balanced.txt", clients)


def convert(imgf, labelf, outf, n):
    f = open(imgf, "rb")
    o = open(outf, "w")
    l = open(labelf, "rb")

    f.read(16)
    l.read(8)
    images = []

    for i in range(n):
        image = [ord(l.read(1))]
        for j in range(28 * 28):
            image.append(ord(f.read(1)))
        images.append(image)

    for image in images:
        o.write(",".join(str(pix) for pix in image) + "\n")
    f.close()
    o.close()
    l.close()


def generate_additive_shares(value, n):
    arr = np.asarray(value)
    rand_arr = np.random.uniform(low=-np.abs(arr), high=np.abs(arr), size=(n - 1,) + arr.shape)
    shares = np.concatenate((rand_arr, [arr - rand_arr.sum(axis=0)]), axis=0)
    return shares


def randomly_select_weights(weights, fraction=0.25):
    # Get number of weights to select
    num_select = int(weights.shape[0] * fraction)

    # Generate random indices to select
    indices = np.random.choice(weights.shape[0], size=num_select, replace=False)

    return indices


def replace_selected(weights, new_weights):
    replaced = []
    for w, nw in zip(weights, new_weights):
        # Indices of weights to replace
        indices = [...]
        # Create copy of existing weights
        updated = np.copy(w)
        # Replace selected indices with new weights
        updated[indices] = nw
        replaced.append(updated)
    return replaced


def exchange_weights(client_weights):
    n = len(client_weights)
    client_shares = []

    # Generate shares for each client's salary
    for salary in client_weights:
        shares = generate_additive_shares(salary, n)
        client_shares.append(shares)

    # Perform secure exchange of shares
    exchanged_shares = [[] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                exchanged_shares[i].append(client_shares[j][i])

        exchanged_shares[i].insert(0, client_shares[i][i])
    return exchanged_shares


def perform_exchange(n, client_shares):
    # Perform secure exchange of shares
    exchanged_shares = [[] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                exchanged_shares[i].append(client_shares[j][i])
        exchanged_shares[i].insert(0, client_shares[i][i])

    return exchanged_shares


def combine_csv_files(experiment, dataset):
    parent_dir = os.path.dirname((os.path.dirname(os.getcwd())))
    folder_path = parent_dir + f'/results/numerical/{experiment}/{dataset}'

    # List all files in the folder
    files = os.listdir(folder_path)

    # Initialize an empty list to store the data frames
    data_frames = []

    # Iterate over each file
    for file in files:
        if file.startswith('client') and file.endswith('.csv'):  # Ensure that only CSV files are considered
            file_path = os.path.join(folder_path, file)

            # Read the CSV file and append it to the list
            df = pd.read_csv(file_path)
            df = df.drop(columns='round')
            data_frames.append(df)

    # Concatenate all data frames into one
    if len(data_frames) != 0:
        combined_df = pd.concat(data_frames, axis=1)
        combined_df.to_csv(f"{folder_path}/combined.csv")

    for file in files:
        if file.startswith('client') and file.endswith('.csv'):
            file_path = os.path.join(folder_path, file)
            os.remove(file_path)

    # return combined_df


def combine_find_mean():
    parent_dir = os.path.dirname((os.path.dirname(os.getcwd())))
    folder_path = parent_dir + f'/results/numerical'

    # folder_items = os.listdir(folder_path)
    folder_items = [
        'additive_rsa'
    ]

    for folder in folder_items:
        for dataset in DATASET:
            combine_csv_files(folder, dataset)

            csv_dir = os.path.join(folder_path, folder, dataset, 'combined.csv')

            if os.path.exists(csv_dir):
                df = pd.read_csv(csv_dir)
                df['Average Accuracy'] = df.apply(lambda row: row['accuracy'].mean(), axis=1)
                df['Average Training'] = df.apply(lambda row: row['training'].mean(), axis=1)
                if 'secret_sharing' in df:
                    df['Average Secret Sharing'] = df.apply(lambda row: row['secret_sharing'].mean(), axis=1)
                df.columns = df.columns.str.replace(r'\.\d+$', '', regex=True)
                df.to_csv(csv_dir, index=False)


def convert_png_to_eps(folder_path):
    files = os.listdir(folder_path)

    for file in files:
        if file.startswith('multiline') and file.endswith('.png'):  # Ensure that only png files are considered
            # file name
            png_file = os.path.join(folder_path, file)

            # Open the PNG file
            image = Image.open(png_file)

            # Create a new EPS file with the same dimensions and mode as the PNG
            eps_image = Image.new("RGB", image.size)
            eps_image.paste(image)

            # Save the EPS file
            eps_image.save(png_file.replace("png", "eps"), format='EPS')


def reconstruct_shamir_secret(shares):
    """
    Combines individual shares (points on graph)
    using Lagranges interpolation.

    `shares` is a list of points (x, y) belonging to a
    polynomial with a constant of our key.
    """
    sums = 0

    for j, share_j in enumerate(shares):
        xj, yj = share_j
        prod = Decimal(1)

        for i, share_i in enumerate(shares):
            xi, _ = share_i
            if i != j:
                prod *= Decimal(Decimal(xi) / (xi - xj))

        prod = yj * np.float64(prod)
        sums = sums + prod

    return np.array(sums, dtype=np.int64)


def polynom(x, coefficients):
    """
    This generates a single point on the graph of given polynomial
    in `x`. The polynomial is given by the list of `coefficients`.
    """
    point = 0
    # Loop through reversed list, so that indices from enumerate match the
    # actual coefficient indices
    for coefficient_index, coefficient_value in enumerate(coefficients[::-1]):
        point = point + x ** coefficient_index * coefficient_value
    return point


def coeff(t, secret):
    """
    Randomly generate a list of coefficients for a polynomial with
    degree of `t` - 1, whose constant is `secret`.

    For example with a 3rd degree coefficient like this:
        3x^3 + 4x^2 + 18x + 554

        554 is the secret, and the polynomial degree + 1 is
        how many points are needed to recover this secret.
        (in this case it's 4 points).
    """
    coeff = [random.randrange(0, FIELD_SIZE) for _ in range(t - 1)]
    coeff.append(secret)
    return coeff


def generate_shamir_shares(n, m, secret):
    """
    Split given `secret` into `n` shares with minimum threshold
    of `m` shares to recover this `secret`, using SSS algorithm.
    """
    coefficients = coeff(m, secret)
    shares = []

    for i in range(1, n + 1):
        x = random.randrange(1, FIELD_SIZE)
        shares.append((x, polynom(x, coefficients)))

    return shares


def generate_groups(num_clients, group_size):
    # shuffle clients
    random.shuffle(list(num_clients))

    # Create a repeating iterator over the clients list
    client_iterator = itertools.cycle(num_clients)

    # Keep track of which clients have been selected
    selected_clients = set()

    # Keep selecting groups of clients until each client has been selected at least once
    selected_groups = []
    while len(selected_clients) < len(num_clients):
        # Select the next group of clients
        group = list(itertools.islice(client_iterator, group_size))

        # Add the selected clients to the set of selected clients
        selected_clients.update(group)

        # Add the selected group to the list of selected groups
        selected_groups.append(group)

    # All clients have been selected at least once
    return selected_groups


class TimingCallback(Callback):
    def __init__(self, logs=None):
        super().__init__()
        if logs is None:
            logs = {}
        self.start_time = None
        self.logs = []

    def on_epoch_begin(self, epoch, logs=None):
        if logs is None:
            logs = {}
        self.start_time = timer()

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        self.logs.append(timer() - self.start_time)


class MemoryPrintingCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        gpu_dict = tf.config.experimental.get_memory_info('GPU:0')
        tf.print('\n GPU memory details [current: {} gb, peak: {} gb]'.format(
            float(gpu_dict['current']) / (1024 ** 3),
            float(gpu_dict['peak']) / (1024 ** 3)))


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, tf.Variable):
            return self.default(obj.numpy())  # Convert tf.Variable to numpy array
        return json.JSONEncoder.default(self, obj)


class NumpyDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.dict_to_object, *args, **kwargs)

    def dict_to_object(self, d):
        if isinstance(d, dict):
            for key, value in d.items():
                if isinstance(value, list):
                    for i, v in enumerate(value):
                        if isinstance(v, list):
                            value[i] = self.list_to_ndarray(v)
                d[key] = value
        return d

    def list_to_ndarray(self, l):
        for i, v in enumerate(l):
            if isinstance(v, list):
                l[i] = self.list_to_ndarray(v)
            elif isinstance(v, str):
                try:
                    l[i] = np.fromstring(v[1:-1], sep=',')
                except:
                    pass
        return np.array(l, dtype=object)


if __name__ == "__main__":
    combine_find_mean()