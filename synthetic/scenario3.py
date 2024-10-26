import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm.auto as tqdm

DISC = True
gamma = 0.9

FILE_PATH = 'synthetic/results/scenario3_disc.npy'
FIGURE_PATH = 'synthetic/figures/scenario3.pdf'

class Data_Scenario3():
    def __init__(self, p=0.9, τ=30, ntest=100):
        self.p = [[p, 1-p], [1-p, p]]
        self.τ = τ
        self.ntest = 50

    def get_samples(self, t):
        y0 = np.random.choice([0, 1])
        y_seq = [y0]
        for _ in range(t + self.ntest - 1):
            y_seq.append(np.random.choice([0, 1], p=self.p[y_seq[-1]]))
        y_train, y_test = y_seq[:t], y_seq[t:]
        return y_train, y_test
    
    def evaluate(self, p_hat, y_test, discount=DISC):

        p_hat = np.array(p_hat)
        y_hat = np.zeros_like(y_test)
        y_hat[p_hat > 0.5] = 1
        y_hat[ np.abs(p_hat - 0.5) < 1e-5] = 0.5

        if not DISC:
            err = np.mean(np.abs(p_hat - y_test))
        else:
            # use discounted sum
            err = 0
            for i in range(len(y_test)):
                err += gamma**i * np.abs(y_hat[i] - y_test[i])
            err = (1 - gamma) * err
        return err

    def estimator_erm(self, samples):
        p_hat = np.mean(samples)
        ntest = self.ntest
        return np.ones(ntest) * p_hat

    def estimator_prospective(self, samples):

        # Pseudo counts (Bayesian prior)
        counts = np.array([[1, 1], [1, 1]])

        for i in range(len(samples)-1):
            counts[samples[i], samples[i+1]] += 1
        probs = counts / np.sum(counts, axis=1, keepdims=True)

        # Estimator 1
        theta = (probs[0, 0] + probs[1, 1]) / 2
        probs = np.array(
            [[theta, 1-theta],
             [1-theta, theta]])
        cur_probs = np.array(probs)

        # Estimator 2
        # cur_probs = np.array(probs)
        
        p_hat = []
        ystart = samples[-1]

        for t in range(self.ntest):
            if np.abs(cur_probs[ystart, 0] - 0.5) < 1e-5:
                ycur = np.random.choice([0, 1])
            else:
                ycur = np.argmax(cur_probs[ystart])
            cur_probs = cur_probs @ probs
            p_hat.append(ycur)

        return p_hat
    
    def get_errs(self, seeds, est=0):
        errs = []
        for t in tqdm.tqdm(range(2, self.τ-1)):
            err_seed = []

            for sd in range(seeds):
                y_train, y_test = dat.get_samples(t)

                if est == 0:
                    p_hat = dat.estimator_erm(y_train)
                else:
                    p_hat = dat.estimator_prospective(y_train)
                err_seed.append(dat.evaluate(p_hat, y_test))
            errs.append(err_seed)
        errs = np.array(errs)
        return errs

def make_plot():
    outputs = np.load(FILE_PATH, allow_pickle=True).tolist()

    labels = [r'MLE', r'Prospective learner']
    keys = ['avg_erm', 'avg_pr']
    cols = ['#377eb8', '#e41a1c']

    plt.figure(figsize=(5, 4))
    for i, label in enumerate(labels):
        plt.plot(np.arange(1, 18, 1), outputs[keys[i]], lw=2.5, marker='o', ms=5, label=label, c=cols[i])
    plt.plot(np.arange(1, 18, 1), 0.357 * np.ones(17), ls='dashed', color='k', lw=2.5, label='Bayes risk')
    plt.locator_params(axis='x', nbins=5)
    plt.locator_params(axis='y', nbins=5)
    plt.xlabel(r"Time ($t$)")
    plt.ylabel(r"Prospective risk")
    plt.legend(frameon=False, fontsize=14, loc='lower right')
    plt.grid(alpha=0.5, ls='--')
    plt.title('Scenario 3' + '\n' + 'Data from a two-state Markov chain', fontsize=14)
    plt.ylim([0.15, 0.55])

    if not os.path.exists(os.path.dirname(FIGURE_PATH)):
        os.makedirs(os.path.dirname(FIGURE_PATH))
    plt.savefig(FIGURE_PATH, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    np.random.seed(0)
    p=0.1
    seeds = 10000
    ntest = 100
    run_t = 20
    times = np.arange(2, run_t-1)

    dat = Data_Scenario3(p, run_t, ntest)
    errs_erm = dat.get_errs(seeds, 0)
    errs_prospective = dat.get_errs(seeds, 1)

    avg_erm = np.mean(errs_erm, axis=1)
    avg_pr = np.mean(errs_prospective, axis=1)
    std_erm = np.std(errs_erm, axis=1)
    std_pr = np.std(errs_prospective, axis=1)

    info = {
        'avg_erm': avg_erm,
        'avg_pr': avg_pr,
        'std_erm': std_erm,
        'std_pr': std_pr,
    }
    if DISC:
        np.save("synthetic/results/scenario3_disc.npy", info, allow_pickle=True)
    else:
        np.save("synthetic/results/scenario3_avg.npy", info, allow_pickle=True)

    make_plot()