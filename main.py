import torch

# To fix the random seed
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

import random

random.seed(0)

import numpy as np
import pandas as pd

np.random.seed(0)

import utils
import os

os.environ["OMP_NUM_THREADS"] = '1'
import warnings

warnings.filterwarnings('ignore')




def main(df):
    args, unknown = utils.parse_args()
    dataset = args.dataset.split("/")[-1]

    from models import scAFGCC_ModelTrainer
    start_memory = utils.show_info()
    embedder = scAFGCC_ModelTrainer(args)
    embedder.train(dataset, df, start_memory)


if __name__ == "__main__":

    df = pd.DataFrame(columns=['dataset', 'total_memory', 'total_time', 'ARI', 'NMI'])
    for i in range(1):
        main(df)
