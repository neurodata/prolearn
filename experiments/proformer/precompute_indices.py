'''
Precompute indices for independent (case 2) exps
'''

from prol.process import (
    get_torch_dataset,
    get_multi_indices_and_map,
    get_multi_sequence_indices,
    get_multi_cycle
)
from datetime import datetime
import pickle
import math
import numpy as np

# specify the task and the experimental details
dataset = 'mnist'
tasks = [
    [0, 1], [1, 0]
]
N = 10
t_list = [0,200,500,700,1000,1200,1500,1700,2000,2500]
T = 5000
initial_seed = 1996
outer_reps = 3
reps = 100

# get the torch dataset
root = '../../data'
torch_dataset, _, _ = get_torch_dataset(root, dataset)

# get the task index dict, label mapper, and updated torch dataset
taskInd, mapdict, torch_dataset = get_multi_indices_and_map(tasks, torch_dataset)
maplab = lambda lab : mapdict[lab]

# get full pattern
unit = get_multi_cycle(N, len(tasks))
full_pattern = np.array((unit * math.ceil(T/(len(unit))))[:T]).astype("int")

# obtain the train/test sequences for the experiment
total_indices = {}
for t in t_list:
    print(f'computing for...t = {t}')
    replicates = []
    for outer_rep in range(outer_reps):
        seed = initial_seed * outer_rep * 2357

        if t > 0:
            train_SeqInd, updated_taskInd = get_multi_sequence_indices(
                N=N, 
                total_time_steps=t, 
                tasklib=taskInd, 
                seed=seed,
                remove_train_samples=True
            )
        else:
            train_SeqInd = []
            updated_taskInd = taskInd

        test_seqInds = [
            get_multi_sequence_indices(N, T, updated_taskInd, seed=seed+1000*(inner_rep+1))
            for inner_rep in range(reps)
        ]
        seq = {}
        seq['train'] = train_SeqInd
        seq['test'] = test_seqInds
        replicates.append(seq)
    total_indices[t] = replicates

total_indices['full_pattern'] = full_pattern
total_indices['task'] = tasks
total_indices['time_list'] = t_list
total_indices['N'] = N
total_indices['T'] = T
total_indices['outer_reps'] = outer_reps
total_indices['inner_reps'] = reps
total_indices['dataset'] = dataset
total_indices['mapdict'] = mapdict

# save the indices
filename = f'{dataset}_{datetime.now().strftime("%H-%M-%S")}'
file = f'indices/{filename}.pkl'
with open(file, 'wb') as f:
    pickle.dump(total_indices, f)