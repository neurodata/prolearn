import os
import numpy as np
import matplotlib
font = {'size':16}
matplotlib.rc('font', **font)
matplotlib.rcParams['figure.facecolor'] = 'white'
import matplotlib.pyplot as plt

FILE_PATH = 'synthetic/results/proMAP.npy'
FIGURE_PATH = 'synthetic/figures/proMAP.pdf'

p=0.9
T=20
T_end=1000
reps=2000
alpha=12
beta=16

def hypothesis(p, size):
    h = np.random.uniform(low=0, high=1, size=size)
    h[p > 0.5, :] = 1
    h[p < 0.5, :] = 0
    return h

def hypothesis_promap(p, size):
    h = np.random.uniform(low=0, high=1, size=size)
    h[p > 0.5] = 1
    h[p < 0.5] = 0
    return h

def loss(truth, pred):
    return (truth - pred)**2

def compute_mle(data):
    return np.mean(data, axis=-1)

def compute_map(data, t, alpha=4, beta=7):
    nom = alpha + np.sum(data, axis=-1) - 1
    denom = alpha + beta + t - 2
    return nom/denom

def compute_delta_dot(p, t, alpha, beta):
    alpha_1 = alpha - 1
    beta_1 = beta - 1
    t_1 = t - 1
    r1 = (alpha_1 + t*p)/(alpha_1 + beta_1 + t)
    r2 = (alpha_1 + t_1*p)/(alpha_1 + beta_1 + t_1)
    return r1 - r2

def compute_prospective_map(p_map, t, T, alpha, beta):
    p = p_map
    delta_dots = [compute_delta_dot(p, r, alpha, beta) for r in np.arange(t, T, 1)]
    int_delta_dots = np.array([np.sum(delta_dots[:r+1], axis=0) for r in range(len(delta_dots))]).T
    p_hat_s = p.reshape(-1, 1) + int_delta_dots
    return p_hat_s

def run():
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
        p_mle = compute_mle(past_data)
        p_map = compute_map(past_data, t, alpha, beta)
        p_promap = compute_prospective_map(p_map, t, T_end, alpha, beta)
        
        # get the hypothesis for each outcome
        h_mle = hypothesis(p_mle, future_data.shape)
        h_map = hypothesis(p_map, future_data.shape)
        h_promap = hypothesis_promap(p_promap, future_data.shape)

        # get the cumulative loss for each outcome
        cumulative_loss_mle = np.mean(loss(future_data, h_mle), axis=-1)
        cumulative_loss_map = np.mean(loss(future_data, h_map), axis=-1)
        cumulative_loss_promap = np.mean(loss(future_data, h_promap), axis=-1)

        # take the weighted average of the cumulative loss over all the outcomes (weight is the probabilities we computed above)
        risk.append(
            [
                np.mean(cumulative_loss_mle),
                np.mean(cumulative_loss_map),
                np.mean(cumulative_loss_promap)
            ]
        )

    risk = np.array(risk)

    case1_outputs = {
        'risk': risk,
        'time': t_list
    }

    if not os.path.exists(os.path.dirname(FILE_PATH)):
        os.makedirs(os.path.dirname(FILE_PATH))
    np.save(FILE_PATH, case1_outputs, allow_pickle=True)

def make_plots():

    outputs = np.load(FILE_PATH, allow_pickle=True).tolist()

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ['#377eb8', '#984ea3', '#e41a1c']

    labels = [
        "MLE", 
        "MAP" + r" (prior $\alpha$ = " + f"{alpha}," + r" $\beta$ = " + f"{beta})",
        "proMAP" + r" (prior $\alpha$ = " + f"{alpha}," + r" $\beta$ = " + f"{beta})",
    ]

    for i, label in enumerate(labels):
        ax.plot(outputs['time'], outputs['risk'][:, i], label=label, marker='o', ms=5, lw=2.5, color=colors[i])

    ax.plot(outputs['time'], (1-p)*np.ones((len(outputs['time']),)), label='Bayes risk', color='k', ls='dashed', lw=2)
    ax.legend(frameon=False, fontsize=14)
    ax.set_ylabel('Prospective risk')
    ax.set_xlabel(r'Time ($t$)')
    ax.set_title('Scenario 1\nIndependent and identically distributed data', fontsize=16, y=1.05)
    ax.set_ylim([0, 1])
    ax.grid(alpha=0.5, ls='--')

    if not os.path.exists(os.path.dirname(FIGURE_PATH)):
        os.makedirs(os.path.dirname(FIGURE_PATH))
    plt.savefig(FIGURE_PATH, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    run()
    make_plots()