# This is a sample Python script.
import os
from code.helpers.constants import NODES, BIT_SIZE
from utils import iid_balanced, generate_keys
from utils import combine_find_mean, convert_png_to_eps

if __name__ == '__main__':
    cwd = os.path.dirname(os.getcwd())

    keys_path = os.path.join(cwd, 'resources', 'keys')
    results_path = os.path.join(cwd, 'resources', 'results')
    plot_path = os.path.join(cwd, 'resources', 'plots')

    # generate encryption keys for all clients
    for i in range(NODES):
        generate_keys(keys_path, 1 + i, nbits=BIT_SIZE)

    # generate IID distribution for all datasets
    iid_balanced(NODES, 50000, "cifar-10")
    iid_balanced(NODES, 60000, "mnist")
    iid_balanced(NODES, 60000, "f-mnist")
    iid_balanced(NODES, 73250, "svhn")

    # run cleaner for dataset and images
    combine_find_mean(results_path)
    convert_png_to_eps(plot_path)
