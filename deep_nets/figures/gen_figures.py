import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns


def make_plot(info, title, figname, size=50, subsample=None,
              outside_legend=False, minimal=False):

    print(title, figname)
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set(context='poster',
            style='ticks',
            font_scale=0.75,
            rc={'axes.grid':True,
                'grid.color':'.9',
                'figure.autolayout': not outside_legend,
                'grid.linewidth':0.75})
    plt.figure(figsize=(5, 5))
    plt.ylim([-0.05, 1])
    plt.title(title)

    cols = ['#377eb8', '#e41a1c', '#4daf4a', '#984ea3',
            '#ff7f00', '#ffff33', '#a65628']

    methods = []
    methods_legend = []
    for m in info:
        methods.append(m)
        if m == 'ERM':
            methods_legend.append('Follow-the-leader')
        else:
            methods_legend.append(m)

    if subsample is not None:
        for m, s in subsample:
            info[methods[m]]= info[methods[m]][:,::s]

    if not minimal:
        plt.ylabel("Prospective Risk")
    plt.xlabel("Time (t)")

    for i, m in enumerate(methods):
        plt.plot(info[m][2], info[m][0], c=cols[i])

    add_bayes = False
    if "_m2" in figname:
        add_bayes = True
        plt.axhline(y=0.2768, color='black', linestyle='--')
    elif not minimal:
        if 'cifar' not in figname and 'mnist' not in figname:
            plt.axhline(y=0.0, color='black', linestyle='--')
            add_bayes = True

    # plot chance risk
    if "syn" in figname:
        plt.axhline(y=0.5, color='#ff7f00', linestyle='--')
    elif "mnist" or "cifar" in figname:
        plt.axhline(y=0.742, color='#ff7f00', linestyle='--')

    for i, m in enumerate(methods):
        plt.scatter(info[m][2], info[m][0], c=cols[i], s=size)
        std = 2 * info[m][1] / np.sqrt(5)
        mean = info[m][0]
        plt.fill_between(info[m][2], mean-std, mean+std,
                         alpha=0.3, color=cols[i])
        
    plt.savefig("./figures/figs/%s.pdf" % figname, bbox_inches='tight')

    if add_bayes:
        methods_legend = methods_legend + ['Bayes risk']

    if not minimal:
        if outside_legend:
            leg = plt.legend(methods_legend,
                       loc="upper right", markerscale=2.,
                       bbox_to_anchor=(1.82, 0.9),
                       scatterpoints=1, fontsize=15, frameon=True)
        else:
            leg = plt.legend(methods_legend,
                       loc="upper right", markerscale=2.,
                       scatterpoints=1, fontsize=15, frameon=True,
                       ncol=len(methods_legend)+1)

    def export_legend(legend, filename="legend.png"):

        learners = [text.get_text() for text in legend.get_texts()][:-1]
        cols = ['#377eb8', '#e41a1c', '#4daf4a', '#984ea3',
                '#ff7f00', '#ffff33', '#a65628']

        fig_legend = plt.figure()

        legend_elements = [
            Line2D([0], [0], color=cols[i], lw=4, label=learner) for i, learner in enumerate(learners)
        ] + [
            Line2D([0], [0], color='k', lw=4, ls='--', label="Bayes Risk"),
            Line2D([0], [0], color='#ff7f00', lw=4, ls='--', label="Chance"),
        ]
        
        fig_legend.legend(
            handles=legend_elements, 
            loc='center', 
            ncol=len(learners)+2, 
            fontsize=15, 
            frameon=True,
            markerscale=2.,
            scatterpoints=1)
        fig_legend.savefig(filename, dpi="figure", bbox_inches='tight')

    if not minimal and not outside_legend:
        export_legend(leg, filename="./figs/aug20/%s_legend.pdf" % figname)


def synthetic_scenario2():
    info = np.load("./metrics/syn_scenario2.pkl", allow_pickle=True)
    make_plot(info, "Synthetic Scenario 2", figname="syn_scenario2")

def synthetic_scenario3():
    info = np.load("./metrics/syn_scenario3.pkl", allow_pickle=True)
    make_plot(info, "Synthetic Scenario 3", figname="syn_scenario3")

def mnist_scenario2():
    info = np.load("./metrics/mnist_scenario2.pkl", allow_pickle=True)
    make_plot(info, "MNIST Scenario 2", figname="mnist_scenario2", minimal=True)

def mnist_scenario3():
    info = np.load("./metrics/mnist_scenario3.pkl", allow_pickle=True)
    make_plot(info, "MNIST Scenario 3", figname="mnist_scenario3", minimal=True)

def cifar_scenario2():
    info = np.load("./metrics/cifar_scenario2.pkl", allow_pickle=True)
    make_plot(info, "CIFAR Scenario 2", figname="cifar_scenario2",
              subsample=[(2, 2), (3, 2)], minimal=True)

def cifar_scenario3():
    info = np.load("./metrics/cifar_scenario3.pkl", allow_pickle=True)
    make_plot(info, "CIFAR Scenario 3", figname="cifar_scenario3", minimal=True)

def synthetic_scenario3_m2():
    info = np.load("./metrics/syn_scenario3_markov2.pkl", allow_pickle=True)
    make_plot(info, "Synthetic Scenario 3", figname="syn_scenario3_m2")

def mnist_scenario3_m2():
    info = np.load("./metrics/mnist_scenario3_markov2.pkl", allow_pickle=True)
    make_plot(info, "MNIST Scenario 3", figname="mnist_scenario3_m2", minimal=True)

def cifar_scenario3_m2():
    info = np.load("./metrics/cifar_scenario3_markov2.pkl", allow_pickle=True)
    make_plot(info, "CIFAR Scenario 3", figname="cifar_scenario3_m2", minimal=True)


synthetic_scenario2()
synthetic_scenario3()
synthetic_scenario3_m2()

mnist_scenario2()
mnist_scenario3()
mnist_scenario3_m2()

cifar_scenario2()
cifar_scenario3()
cifar_scenario3_m2()
