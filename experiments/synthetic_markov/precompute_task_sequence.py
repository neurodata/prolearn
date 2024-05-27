'''
Precompute task seuqnece for synthetic Markov (case 3) exps
'''

from prol.process import (
    get_markov_chain
)
from datetime import datetime
import pickle
import numpy as np

# specify the task and the experimental details
dataset = 'synthetic'
N = 20
T = 5000
initial_seed = 1996
outer_reps = 50

num_tasks = 2
patterns = np.array([get_markov_chain(num_tasks, T, N, seed=k*11111) for k in range(outer_reps)])

output = {
    "pattern": patterns,
    "N": N,
    "T": T,
    "outer_reps": outer_reps
}

fname = f'{dataset}_{outer_reps}.pkl'
with open(fname, 'wb') as f:
    pickle.dump(output, f)



