import pickle
import os
import numpy as np
from functools import partial

def get_data_split(data_split_file, hmr_dir):
    if os.path.exists(data_split_file):
        with open(data_split_file, 'r') as f:
            examples = f.read()
        trn_ex, tst_ex = examples.split('|')
        trn_ex = trn_ex.split(',')
        tst_ex = tst_ex.split(',')
    else:
        examples = os.listdir(hmr_dir)
        np.random.shuffle(examples)
        num_trn = int(0.9 * len(examples))
        trn_ex = examples[:num_trn]
        tst_ex = examples[num_trn:]
        examples = '|'.join([','.join(i) for i in [trn_ex, tst_ex]])
        with open(data_split_file, 'w') as f:
            f.write(examples)
    return trn_ex, tst_ex

def get_example(example, hmr_dir):
    with open(os.path.join(hmr_dir, example), 'rb') as f:
        example = pickle.load(f)
    return example

def load_all_examples(examples, hmr_dir):
    '''

    Parameters
    ----------
    examples : list
        list of lists of example names

    '''
    return [np.stack(list(
        map(partial(get_example, hmr_dir=hmr_dir), i)), 0) for i in examples]

def get_norm_params(examples):
    return np.mean(examples, axis=(0,1)), np.std(examples, axis=(0,1))

def load_normalised_examples(data_split_file, hmr_dir):
    trn_ex, tst_ex = load_all_examples(
        get_data_split(
            data_split_file=data_split_file,
            hmr_dir=hmr_dir),
        hmr_dir=hmr_dir)
    mean, std = get_norm_params(trn_ex)
    trn_ex = (trn_ex - mean)/std
    tst_ex = (tst_ex - mean)/std
    return trn_ex, tst_ex