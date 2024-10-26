import numpy as np
import matplotlib
font = {'size':16}
matplotlib.rc('font', **font)
matplotlib.rcParams['figure.facecolor'] = 'white'
import matplotlib.pyplot as plt

from scenario4 import get_data

class Data_Scenario3():
    def __init__(self, p=0.9, τ=30, max_t=1000):
        self.p = [[p, 1-p], [1-p, p]]
        self.τ = τ
        self.max_t = max_t

    def get_samples(self, t):
        y0 = 1
        y_seq = [y0]
        for _ in range(t + 10 - 1):
            y_seq.append(np.random.choice([0, 1], p=self.p[y_seq[-1]]))
        return y_seq

np.random.seed(1996)

fig, ax = plt.subplots(1, 4 ,figsize=(15, 4))
fontsize = 14
marker_size = 7
p = 0.8
q = 0.2
T = 20
T_initial = 10
T_end = 20
reps_1 = 51
reps_2 = 100
initial_show = int(reps_1/2)

## Senario 1
outcomes = np.random.binomial(1, p, size=(reps_1 , T_end))
outcomes[:initial_show,:T_initial] = np.zeros((initial_show,T_initial))
outcomes[initial_show+1:,:T_initial] = np.zeros((reps_1-1-initial_show,T_initial))
t_list = np.arange(1, T+1, 1)
outcomes = outcomes.astype(float)
outcomes[outcomes==0] = np.nan
for rep in range(reps_1):
    y = (outcomes[rep,:]+rep) * outcomes[rep,:]
    # print(y)
    if rep != initial_show:
        y[:T_initial] = np.nan
    ax[0].scatter(t_list,y, c = 'black',alpha = 0.8,s = marker_size)

### Senario 2
pattern = np.array([True, False]*T_end)[:T_end]
p_pattern = np.zeros(T_end)
p_pattern[pattern] = p
p_pattern[~pattern] = q
outcomes = np.random.binomial(1, p_pattern, size=(reps_1, T_end))
outcomes[:initial_show,:T_initial] = np.zeros((initial_show,T_initial))
outcomes[initial_show+1:,:T_initial] = np.zeros((reps_1-1-initial_show,T_initial))
outcomes = outcomes.astype(float)
outcomes[outcomes==0] = np.nan
t_list = np.arange(1, T+1, 1)
for rep in range(reps_1):
    y = (outcomes[rep,:]+rep) * outcomes[rep,:]
    if rep != initial_show:
        y[:T_initial] = np.nan
    ax[1].scatter(t_list,y, c = 'black',alpha = 0.8,s = marker_size)

### Senario 3
t_list = np.arange(1, T+1, 1)
outcomes = np.zeros((reps_1,T))
for i in range(reps_1):
    outcomes_i = Data_Scenario3().get_samples(t = 10)
    outcomes[i,:] = outcomes_i
outcomes[:initial_show,:T_initial] = np.zeros((initial_show,T_initial))
outcomes[initial_show+1:,:T_initial] = np.zeros((reps_1-1-initial_show,T_initial))
outcomes = outcomes.astype(float)
outcomes[outcomes==0] = np.nan
for rep in range(reps_1):
    y = (outcomes[rep,:]+rep) * outcomes[rep,:]
    if rep != initial_show:
        y[:T_initial] = np.nan
    ax[2].scatter(t_list,y, c = 'black',alpha = 0.8,s = marker_size)

### Scenario 4
t_list = np.arange(1, T+1, 1)
outcomes = np.zeros((reps_1,T))
for i in range(reps_1):
    z, _, _ = get_data(20)
    outcomes_i = z
    outcomes[i,:] = outcomes_i
outcomes[:initial_show,:T_initial] = np.zeros((initial_show,T_initial))
outcomes[initial_show+1:,:T_initial] = np.zeros((reps_1-1-initial_show,T_initial))
outcomes = outcomes.astype(float)
outcomes[outcomes==0] = np.nan
for rep in range(reps_1):
    y = (outcomes[rep,:]+rep) * outcomes[rep,:]
    if rep != initial_show:
        y[:T_initial] = np.nan
    ax[3].scatter(t_list,y, c = 'black',alpha = 0.8,s = marker_size)

### Plot
for i in range(4):
    ax[i].set_yticks([])
    ax[i].set_xticks([0,10],[0,'t'])
    ax[i].set_xlabel('Time',fontsize = fontsize)

    ax[i].spines['top'].set_visible(False)
    ax[i].spines['left'].set_visible(False)
    ax[i].spines['right'].set_visible(False)
    ax[i].spines['bottom'].set_linewidth(2)

    ax[i].tick_params(axis='x', labelsize=fontsize, length=8, width=2)

    ax[i].axvline(x=10, color='#D3D3D3', linestyle='--', linewidth=2)

    ax[i].text(3, 55, 'Realized\nPast', fontsize=12)
    ax[i].text(13, 55, 'Potential\nFutures', fontsize=12)
    ax[i].text(7.5, 63, f'Scenario {i+1}', fontsize=12)

    if i ==0:
        ax[i].set_ylabel('Trials',fontsize = fontsize)
    else:
        ax[i].set_ylabel('')

plt.savefig("synthetic/figures/rastergram.pdf", format="pdf", bbox_inches="tight") 