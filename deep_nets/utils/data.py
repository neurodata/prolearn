import numpy as np
import pickle
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import transforms


class Scenario2:
    def __init__(self, cfg):
        self.seq_len = cfg.seq_len
        self.num_seeds = cfg.num_seeds
        self.period = cfg.period
        self.cfg = cfg

    def generate_data(self):
        xseq, yseq, taskseq = [], [], []
        tseq = []
        for sd in range(self.num_seeds):
            dat = self.gen_sequence(sd)
            xseq.append(dat[0])
            yseq.append(dat[1])
            taskseq.append(dat[2])
            tseq.append(np.arange(self.seq_len))

        xseq = np.array(xseq) / 4
        yseq = np.array(yseq)
        tseq = np.array(tseq)
        taskseq = np.array(taskseq)

        self.data = {'x': xseq,
                     'y': yseq,
                     't': tseq,
                     'task': taskseq,
                     'cfg': self.cfg}


class Scenario3:
    def __init__(self, cfg):
        self.seq_len = cfg.seq_len
        self.num_seeds = cfg.num_seeds
        self.period = cfg.period
        self.cfg = cfg
        self.variant = cfg.variant

    def generate_data(self):
        xseq, yseq, taskseq = [], [], []
        tseq = []
        for sd in range(self.num_seeds):
            # Markov chain with 2 states, reset ever K steps
            if self.variant == 'markov2':
                dat = self.gen_sequence_markov2(sd, stationary=False)
            # Markov chain that equilibriates to the stationary distribution
            if self.variant == 'markov2_s':
                dat = self.gen_sequence_markov2(sd, stationary=True)
            # Markov chain with 4 states (described in the paper)
            elif self.variant == 'markov4':
                dat = self.gen_sequence_markov4(sd)
            else:
                raise ValueError('Invalid variant')
            xseq.append(dat[0])
            yseq.append(dat[1])
            taskseq.append(dat[2])
            tseq.append(np.arange(self.seq_len))

        xseq = np.array(xseq)
        yseq = np.array(yseq)
        tseq = np.array(tseq)
        taskseq = np.array(taskseq)

        self.data = {'x': xseq,
                     'y': yseq,
                     't': tseq,
                     'task': taskseq,
                     'cfg': self.cfg}


class SyntheticScenario2(Scenario2):
    """
    Create multiple sequences for Scenario 2
    """
    def gen_sequence(self, seed):
        np.random.seed(seed)

        # generate a samples from from U[-2, -1] union U[2, 1]
        x1 = np.random.uniform(-2, -1, self.seq_len)
        x2 = np.random.uniform(1, 2, self.seq_len)
        mask = np.random.choice([0, 1], p=[0.5, 0.5], size=self.seq_len)
        Xdat = x1 * mask + x2 * (1 - mask)

        # Create labels
        T = self.period
        tind = np.array((np.arange(0, self.seq_len) % T) < (T // 2))
        tind = tind.astype(int)
        ind = np.where(tind == 1)[0]

        Ydat = Xdat > 0
        Ydat[ind] = Xdat[ind] < 0
        Ydat = Ydat.astype(int)
        
        Xdat = Xdat.reshape(-1, 1)

        return Xdat, Ydat, tind

    def store_data(self):
        os.makedirs('data/synthetic', exist_ok=True)
        with open('data/synthetic/scenario2_period%d.pkl' % self.period, 'wb') as fp:
            pickle.dump(self.data, fp)


class SyntheticScenario3(Scenario3):
    """
    Generate data from a markov process
    """
    def gen_sequence_markov4(self, seed):
        np.random.seed(seed)

        # create task indices
        T = self.period
        cur_t = 0
        tind = []
        for i in range(self.seq_len):
            tind.append(cur_t)

            # Every T steps, switch task
            if (i + 1) % T == 0:
                cur_t = 0
            elif (i + 1) % (T // 2) == 0:
                cur_t = 1

            if cur_t <= 1:
                # Change task with probability 0.2
                if np.random.rand() < 0.2:
                    cur_t = 2 - cur_t
            else:
                # Change task with probability 0.2
                if np.random.rand() < 0.2:
                    cur_t = 4 - cur_t
        tind = np.array(tind)

        # generate a samples from U[1, 2]
        x1 = np.random.uniform(1, 2, self.seq_len)
        x2 = np.random.uniform(1, 2, self.seq_len)
        Xdat = np.stack([x1, x2]).T

        # Generate labels 
        Ydat = np.random.choice([0, 1], size=self.seq_len)

        # Generate data points
        tind_m = (tind + Ydat) % 4

        xmask1 = 1 - (tind_m < 2) * 2
        xmask2 = 1 - (tind_m >= 1) * (tind_m <= 2) * 2

        xmask = np.stack([xmask1, xmask2]).T
        Xdat = Xdat * xmask

        return Xdat, Ydat, tind

    def gen_sequence_markov2(self, seed, stationary=False):
        np.random.seed(seed)
    
        # Create task indices
        T = self.period
        cur_t = 0
        tind = []
        for i in range(self.seq_len):
            tind.append(cur_t)
    
            # Every T steps, switch task
            if (not stationary) and (i + 1) % T == 0:
                cur_t = 0
            else:
                # Change task with probability 0.9
                if np.random.rand() < 0.9:
                    cur_t = 1 - cur_t
    
        tind = np.array(tind)
    
        # Generate samples from U[-2, -1] and U[1, 2]
        x1 = np.random.uniform(-2, -1, self.seq_len)
        x2 = np.random.uniform(1, 2, self.seq_len)
        mask = np.random.choice([0, 1], p=[0.5, 0.5], size=self.seq_len)
        Xdat = x1 * mask + x2 * (1 - mask)
    
        # Create labels
        Ydat = np.zeros(self.seq_len, dtype=int)
        Ydat[tind == 0] = (Xdat[tind == 0] > 0).astype(int)
        Ydat[tind == 1] = (Xdat[tind == 1] < 0).astype(int)
    
        Xdat = Xdat.reshape(-1, 1)

        return Xdat, Ydat, tind

    def store_data(self):
        os.makedirs('data/synthetic', exist_ok=True)
        with open('data/synthetic/scenario3_%s.pkl' % self.variant, 'wb') as fp:
            pickle.dump(self.data, fp)


class MnistScenario2(Scenario2):
    def __init__(self, cfg):
        super(MnistScenario2, self).__init__(cfg)
        self.dataset = MNIST(root='data', train=True, download=True,
                           transform=transforms.ToTensor())

        get_ind = []
        targets = np.array(self.dataset.targets)
        for i in range(10):
            get_ind.append(np.where(targets == i)[0])
        self.yind = get_ind

    def gen_sequence(self, seed):
        np.random.seed(seed)

        # task 1 - {0, 1, 2, 3, 4}
        # task 2- {3, 4, 5, 6}
        # task 3 - {5, 6, 7, 8}
        # task 4 - {7, 8, 9}

        # create task indices
        cur_t = 0
        tind = []
        xseq = []
        yseq = []
        for i in range(self.seq_len):
            cur_t = i % 4
            tind.append(cur_t)

            if cur_t == 0:
                y = np.random.randint(0, 5)
                yseq.append(y - 0)
            elif cur_t == 1:
                y = np.random.randint(3, 7)
                yseq.append(y - 3)
            elif cur_t == 2:
                y = np.random.randint(5, 9)
                yseq.append(y - 5)
            elif cur_t == 3:
                y = np.random.randint(7, 10)
                yseq.append(y - 7)

            xind = np.random.choice(self.yind[y])
            xseq.append(self.dataset.data[xind].reshape(-1) / 255.0)

        Xdat = np.stack(xseq)
        Ydat = np.array(yseq)
        tind = np.array(tind)
        return Xdat, Ydat, tind

    def store_data(self):
        os.makedirs('data/mnist', exist_ok=True)
        with open('data/mnist/scenario2.pkl', 'wb') as fp:
            pickle.dump(self.data, fp)


class MnistScenario3(Scenario3):
    def __init__(self, cfg):
        super(MnistScenario3, self).__init__(cfg)
        self.dataset = MNIST(root='data', train=True, download=True,
                           transform=transforms.ToTensor())

        get_ind = []
        targets = np.array(self.dataset.targets)
        for i in range(10):
            get_ind.append(np.where(targets == i)[0])
        self.yind = get_ind

    def gen_sequence_markov2(self, seed, stationary=False):
        np.random.seed(seed)

        # task 1 - {0, 1, 2, 3, 4}
        # task 2- {4, 0, 1, 2, 3}

        # create task indices
        T = self.period
        cur_t = 0
        xseq = []
        yseq = []
        tind = []
        for i in range(self.seq_len):
            tind.append(cur_t)
            if cur_t == 0:
                y = np.random.randint(0, 5)
                yseq.append(y - 0)
            elif cur_t == 1:
                y = np.random.randint(0, 5)
                yseq.append((y + 1) % 5)

            xind = np.random.choice(self.yind[y])
            xseq.append(self.dataset.data[xind].reshape(-1) / 255.5)

            # Every T steps, switch task
            if (not stationary) and (i + 1) % T == 0:
                cur_t = 0
            elif np.random.rand() < 0.9:
                cur_t = 1 - cur_t

        Xdat = np.stack(xseq)
        Ydat = np.array(yseq)
        tind = np.array(tind)

        return Xdat, Ydat, tind

    def gen_sequence_markov4(self, seed):
        np.random.seed(seed)

        # task 1 - {0, 1, 2, 3, 4}
        # task 2- {3, 4, 5, 6}
        # task 3 - {5, 6, 7, 8}
        # task 4 - {7, 8, 9}

        # create task indices
        T = self.period
        cur_t = 0
        xseq = []
        yseq = []
        tind = []
        for i in range(self.seq_len):
            tind.append(cur_t)
            if cur_t == 0:
                y = np.random.randint(0, 5)
                yseq.append(y - 0)
            elif cur_t == 1:
                y = np.random.randint(3, 7)
                yseq.append(y - 3)
            elif cur_t == 2:
                y = np.random.randint(5, 9)
                yseq.append(y - 5)
            elif cur_t == 3:
                y = np.random.randint(7, 10)
                yseq.append(y - 7)

            xind = np.random.choice(self.yind[y])
            xseq.append(self.dataset.data[xind].reshape(-1) / 255.5)

            # Every T steps, switch task
            if (i + 1) % T == 0:
                cur_t = 0
            elif (i + 1) % (T // 2) == 0:
                cur_t = 1

            if cur_t <= 1:
                # Change task with probability 0.2
                if np.random.rand() < 0.2:
                    cur_t = 2 - cur_t
            else:
                # Change task with probability 0.2
                if np.random.rand() < 0.2:
                    cur_t = 4 - cur_t
        Xdat = np.stack(xseq)
        Ydat = np.array(yseq)
        tind = np.array(tind)

        return Xdat, Ydat, tind

    def store_data(self):
        os.makedirs('data/synthetic', exist_ok=True)
        with open('data/mnist/scenario3_%s.pkl' % self.variant, 'wb') as fp:
            pickle.dump(self.data, fp)


class CifarScenario2(MnistScenario2):
    def __init__(self, cfg):
        self.seq_len = cfg.seq_len
        self.num_seeds = cfg.num_seeds
        self.period = cfg.period
        self.cfg = cfg

        self.dataset = CIFAR10(root='data', train=True, download=True,
                               transform=transforms.ToTensor())

        get_ind = []
        targets = np.array(self.dataset.targets)
        for i in range(10):
            get_ind.append(np.where(targets == i)[0])
        self.yind = get_ind

    def store_data(self):
        os.makedirs('data/cifar', exist_ok=True)
        with open('data/cifar/scenario2.pkl', 'wb') as fp:
            pickle.dump(self.data, fp)


class CifarScenario3(MnistScenario3):
    """
    Generate data from a markov process
    """
    def __init__(self, cfg):
        super(CifarScenario3, self).__init__(cfg)

        self.dataset = CIFAR10(root='data', train=True, download=True,
                               transform=transforms.ToTensor())

        get_ind = []
        targets = np.array(self.dataset.targets)
        for i in range(10):
            get_ind.append(np.where(targets == i)[0])
        self.yind = get_ind

    def store_data(self):
        os.makedirs('data/cifar', exist_ok=True)
        with open('data/cifar/scenario3_%s.pkl' % self.variant, 'wb') as fp:
            pickle.dump(self.data, fp)


class SyntheticDataset(Dataset):
    def __init__(self, data, idx, run_id, test, past=None):
        self.x = data['x']
        self.y = data['y']
        self.t = data['t']

        if test:
            self.x = self.x[run_id, :]
            self.y = self.y[run_id, :]
            self.t = self.t[run_id, :]
        else:
            if past is None:
                # Create dataloader with full history
                self.x = self.x[run_id, :idx]
                self.y = self.y[run_id, :idx]
                self.t = self.t[run_id, :idx]
            else:
                # Create dataloader with just recent history
                self.x = self.x[run_id, idx-past:idx]
                self.y = self.y[run_id, idx-past:idx]
                self.t = self.t[run_id, idx-past:idx]
       
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        t = self.t[idx]
        return x, y, t


def create_dataloader(cfg, t, seed, data):
    past = cfg.fine_tune

    train_dataset = SyntheticDataset(data, t, seed, False, past)
    test_dataset = SyntheticDataset(data, t, seed, True, past)

    trainloader = DataLoader(train_dataset,
                            batch_size=cfg.data.bs,
                            shuffle=True, pin_memory=True,
                            num_workers=cfg.data.workers)
    testloader = DataLoader(test_dataset, batch_size=500,
                            shuffle=False, pin_memory=True,
                            num_workers=cfg.data.workers)

    return trainloader, testloader

