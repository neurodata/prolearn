import os
import numpy as np
import matplotlib
font = {'size':16}
matplotlib.rc('font', **font)
matplotlib.rcParams['figure.facecolor'] = 'white'
import matplotlib.pyplot as plt
import seaborn as sns

import itertools
from scipy.stats import bootstrap
from scipy.stats import beta as beta_dist
from joblib import Parallel, delayed
from scipy.special import beta as B
from scipy.stats import bernoulli

FILE_PATH = 'binary/results/scenario2.npy'
FIGURE_PATH = 'binary/figures/scenario2.pdf'

def loss(truth, pred):
    return (truth - pred)**2

def compare_with_tolerance(num1, num2, atol=1e-9, rtol=1e-9):
    """
    Compare two numbers with numerical tolerance using numpy.isclose().
    Parameters:
        num1 (float): First number
        num2 (float): Second number
        atol (float): Absolute tolerance value for numerical comparison
        rtol (float): Relative tolerance value for numerical comparison
    Returns:
        int: 1 if num1 > num2, -1 if num1 < num2, 0 if num1 and num2 are within tolerance
    """
    if np.isclose(num1, num2, atol=atol, rtol=rtol):
        return 0
    elif num1 > num2 and not np.isclose(num1, num2, atol=atol, rtol=rtol):
        return 1
    elif num1 < num2 and not np.isclose(num1, num2, atol=atol, rtol=rtol):
        return -1
    else:
        return 0

def hypothesis(p, size):
    h = np.random.uniform(low=0, high=1, size=size)
    h[p >= 0.5] = 1
    h[p < 0.5] = 0
    return h.astype('float')

def compute_mle(data):
    return np.mean(data, axis=-1)

def term_hypothesis(phat, qhat, pattern, t, T):
    h = np.random.uniform(low=0, high=1, size=T-t)
    theta = np.zeros((T-t,))
    mask = pattern[t:T]
    theta[mask] = phat
    theta[~mask] = qhat
    h[theta > 0.5] = 1
    h[theta < 0.5] = 0
    return h.astype('float')

def compute_term_mle(past_data, pattern, type=1):
    if type == 1:
        data = np.copy(past_data)
        t = len(data)
        mask = pattern[:t]
        phat = data[mask].mean() if t==1 else (data[mask].mean() + (1 - data[~mask].mean()))/2
        return phat, 1-phat
    elif type == 2:
        data = np.copy(past_data)
        t = len(data)
        mask = np.copy(pattern[:t])
        phat = data[mask].mean()
        qhat = 0.5 if t==1 else data[~mask].mean()
        return phat, qhat
    else:
        data = np.copy(past_data)
        T = len(pattern)
        t = len(data)

        # ABAB
        pattern1 = pattern
        mask1 = pattern1[:t]

        data1 = np.copy(data)
        phat1 = data[mask1].mean()
        qhat1 = 0.5 if len(data[~mask1]) == 0 else data[~mask1].mean()

        L1 = bernoulli.logpmf(data1[mask1], phat1).sum() + bernoulli.logpmf(data1[~mask1], qhat1).sum()

        # AABBAABB
        pattern2 = np.array([True, True, False, False]*T)[:T]
        mask2 = pattern2[:t]

        data2 = np.copy(data)
        phat2 = data[mask2].mean()
        qhat2 = 0.5 if len(data[~mask1]) == 0 else data[~mask2].mean()

        L2 = bernoulli.logpmf(data2[mask2], phat2).sum() + bernoulli.logpmf(data2[~mask2], qhat2).sum()

        flag = compare_with_tolerance(L1, L2, atol=1e-9, rtol=1e-9)
        if flag == 1:
            return phat1, qhat1, pattern1
        elif flag == -1:
            return phat2, qhat2, pattern2
        else:
            return phat1, qhat1, pattern1
        
def run(p=0.8, q=0.2, T=20, T_end=1000, reps=10000):
    # ABAB pattern
    pattern = np.array([True, False]*T_end)[:T_end]

    # sample a bunch of sequences from the process
    p_pattern = np.zeros(T_end)
    p_pattern[pattern] = p
    p_pattern[~pattern] = q
    outcomes = np.random.binomial(1, p_pattern, size=(reps, T_end))

    t_list = np.arange(1, T, 1)
    risk = []

    def loop(i, t, outcomes, pattern):
        past_data = outcomes[i, :t]
        future_data = outcomes[i, t:]
        losses = []

        # naive MLE
        p_hat = compute_mle(past_data)
        h = hypothesis(p_hat, future_data.shape)
        losses.append(np.mean(loss(future_data, h)))

        # MLE 2
        p_hat, q_hat = compute_term_mle(past_data, pattern, type=2)
        h = term_hypothesis(p_hat, q_hat, pattern, t, T_end)
        losses.append(np.mean(loss(future_data, h)))
        
        # MLE 1
        p_hat, q_hat = compute_term_mle(past_data, pattern, type=1)
        h = term_hypothesis(p_hat, q_hat, pattern, t, T_end)
        losses.append(np.mean(loss(future_data, h)))

        # # MLE 3
        p_hat, q_hat, pred_pattern = compute_term_mle(past_data, pattern, type=3)
        h = term_hypothesis(p_hat, q_hat, pred_pattern, t, T_end)
        losses.append(np.mean(loss(future_data, h)))
        return losses

    for t in t_list:
        tmp = Parallel(n_jobs=-1)(delayed(loop)(i, t, outcomes, pattern) for i in range(len(outcomes)))
        risk.append(np.mean(tmp, axis=0))
    risk = np.array(risk)

    case2_outputs = {
        "risk": risk, 
        "time": t_list
    }

    if not os.path.exists(os.path.dirname(FILE_PATH)):
        os.makedirs(os.path.dirname(FILE_PATH))
    np.save(FILE_PATH, case2_outputs, allow_pickle=True)

def make_plot():
    outputs = np.load(FILE_PATH, allow_pickle=True).tolist()

    labels = [r'MLE', r'Prospective learner', r'Variant 2', r'Variant 3']
    cols = ['#377eb8', '#e41a1c', '#984ea3', '#ff7f00']

    # swap lines:
    tmp = np.array(outputs['risk'][:, 1])
    outputs['risk'][:, 1] = outputs['risk'][:, 3]
    outputs['risk'][:, 3] = tmp

    plt.figure(figsize=(5, 4))
    for i, label in enumerate(labels):
        plt.plot(outputs['time'], outputs['risk'][:, i], lw=2.5, marker='o', ms=5, label=label, c=cols[i])
    plt.plot(outputs['time'], 0.2 * np.ones(len(outputs['time'])), ls='dashed', color='k', lw=2.5, label='Bayes risk')
    plt.locator_params(axis='x', nbins=5)
    plt.xlabel(r"Time ($t$)")
    plt.ylabel(r"Prospective risk")
    plt.legend(frameon=False, fontsize=14, loc=0, bbox_to_anchor=(0.3, 0.55))
    plt.grid(alpha=0.5, ls='--')
    plt.title('Scenario 2' + '\n' + 'Independent but not identically distributed data', fontsize=12)

    plt.ylim([0.15, 0.55])

    if not os.path.exists(os.path.dirname(FIGURE_PATH)):
        os.makedirs(os.path.dirname(FIGURE_PATH))
    plt.savefig(FIGURE_PATH, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    run()
    make_plot()