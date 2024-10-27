import os
import numpy as np
import matplotlib
font = {'size':16}
matplotlib.rc('font', **font)
matplotlib.rcParams['figure.facecolor'] = 'white'
import matplotlib.pyplot as plt
import seaborn as sns

FILE_PATH = 'binary/results/scenario1.npy'
FIGURE_PATH = 'binary/figures/scenario1.pdf'

def hypothesis(p, size):
    h = np.random.uniform(low=0, high=1, size=size)
    h[p > 0.5, :] = 1
    h[p < 0.5, :] = 0
    return h

def loss(truth, pred):
    return (truth - pred)**2

def compute_mle(data):
    return np.mean(data, axis=-1)

def run(p=0.8, T=20, T_end=1000, reps=10000):
    # sample a bunch of sequences from the process
    outcomes = np.random.binomial(1, p, size=(reps, T_end))

    t_list = np.arange(1, T, 1)
    risk = []

    # loop over increasing values of t
    for t in t_list:

        # get the past (training) data at each outcome
        past_data = outcomes[:, :t]

        # get the future (evaluation) data at each outcome
        future_data = outcomes[:, t:]

        # compute the p_hat at each outcome based on past data
        p_hat = compute_mle(past_data)
        
        # get the hypothesis for each outcome
        h = hypothesis(p_hat, future_data.shape)

        # get the cumulative loss for each outcome
        cumulative_loss = np.mean(loss(future_data, h), axis=-1)

        # take the weighted average of the cumulative loss over all the outcomes (weight is the probabilities we computed above)
        expected_cumulative_loss = np.mean(cumulative_loss)
        risk.append(expected_cumulative_loss)

    case1_outputs = {
        'risk': risk,
        'time': t_list
    }

    if not os.path.exists(os.path.dirname(FILE_PATH)):
        os.makedirs(os.path.dirname(FILE_PATH))
    np.save(FILE_PATH, case1_outputs, allow_pickle=True)

def make_plot():

    outputs = np.load(FILE_PATH, allow_pickle=True).tolist()

    plt.figure(figsize=(5, 4))
    plt.plot(outputs['time'], outputs['risk'], lw=2.5, marker='o', ms=5, label='MLE', color='#e41a1c')
    plt.plot(outputs['time'], 0.2 * np.ones(len(outputs['time'])), ls='dashed', color='k', lw=2.5, label='Bayes risk')
    plt.locator_params(axis='x', nbins=5)
    plt.xlabel(r"Time ($t$)")
    plt.ylabel(r"Prospective risk")
    plt.legend(frameon=False, fontsize=14)
    plt.grid(alpha=0.5, ls='--')
    plt.title('Scenario 1' + '\n' + 'Independent and identically distributed data', fontsize=12)
    plt.ylim([0.15, 0.55])

    if not os.path.exists(os.path.dirname(FIGURE_PATH)):
        os.makedirs(os.path.dirname(FIGURE_PATH))
    plt.savefig(FIGURE_PATH, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    run()
    make_plot()
