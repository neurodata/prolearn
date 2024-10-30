###### Synthetic data
python3 02_train.py net.type=mlp3 name='erm_mlp' tag=scenario3 \
    tstart=50 tend=4000 tskip=250 \
    train.epochs=100 data.bs=32 \
    data.path='./data/synthetic/scenario3_markov4.pkl'

python3 02_train.py  net.type=prospective_mlp3 name='prospective_mlp' tag=scenario3 \
    tstart=50 tend=4000 tskip=200 \
    train.epochs=100 data.bs=32 \
    data.path='./data/synthetic/scenario3_markov4.pkl'

python3 02_train.py net.type=mlp3 name='mlp_ft1' tag=scenario3 \
    tstart=50 tskip=250 tend=4000  \
    train.epochs=1 fine_tune=16 data.bs=8 \
    data.path='./data/synthetic/scenario3_markov4.pkl'

python3 02_train.py net.type=mlp3 name='mlp_bgd' tag=scenario3 \
    tstart=50 tskip=250 tend=4000 \
    train.epochs=1 fine_tune=16 data.bs=8 bgd=True \
    data.path='./data/synthetic/scenario3_markov4.pkl'

python3 02_train.py  name='prospective_mlp' tag=scenario3_markov2 \
    tstart=50 tend=10051 tskip=500  \
    net.type=prospective_mlp \
    train.epochs=100 data.bs=32 \
    data.path='./data/synthetic/scenario3_markov2.pkl'

python3 02_train.py name='erm_mlp' tag=scenario3_markov2 \
    tstart=50 tend=10051 tskip=500 \
    train.epochs=100 data.bs=32 net.type=mlp \
    data.path='./data/synthetic/scenario3_markov2.pkl'

### MNIST
python3 02_train.py net.type=mlp_mnist name='erm_mlp' \
    tag=mnist_s3 numseeds=5 tstart=20 tend=5021 tskip=250 \
    train.epochs=100 data.bs=32 data.path='./data/mnist/scenario3_markov4.pkl'

python3 02_train.py net.type=prospective_mlp_mnist name='prospective_mlp' \
    tag=mnist_s3 numseeds=5 tstart=20 tend=5021 tskip=250 \
    train.epochs=100 data.bs=32 data.path='./data/mnist/scenario3_markov4.pkl'

python3 02_train.py net.type=mlp_mnist name='mlp_ft1' tag=mnist_s3_v2 \
    tstart=20 tend=5021 tskip=250 \
    train.epochs=1 fine_tune=16 data.bs=8 data.path='./data/mnist/scenario3_markov4.pkl'

python3 02_train.py net.type=mlp_mnist name='mlp_bgd' tag=mnist_s3 \
    tstart=20 tend=5021 tskip=250 \
    train.epochs=1 fine_tune=16 data.bs=8 bgd=True data.path='./data/mnist/scenario3_markov4.pkl'

python3 02_train.py net.type=mlp_mnist name='erm_mlp' \
    tag=mnist_s3_markov2 tstart=20 tend=10021 tskip=500 \
    train.epochs=100 data.bs=32 data.path='./data/mnist/scenario3_markov2.pkl'

python3 02_train.py net.type=prospective_mlp_mnist name='prospective_mlp' \
    tag=mnist_s3_markov2 tstart=20 tend=10021 tskip=500 \
    train.epochs=100 data.bs=32 data.path='./data/mnist/scenario3_markov2.pkl'

### CIFAR
python3 02_train.py  net.type=cnn_cifar name='erm_cnn' tag=cifar_s3 \
    tstart=20 tend=30021 tskip=1500 \
    train.epochs=100 data.bs=32 data.path='./data/cifar/scenario3_markov4.pkl'

python3 02_train.py  net.type=prospective_cnn_cifar name='prospective_cnn_o'  tag=cifar_s3 \
    tstart=20 tend=30021 tskip=1500 \
    train.epochs=100 data.bs=32 data.path='./data/cifar/scenario3_markov4.pkl'

python3 02_train.py  net.type=cnn_cifar name='cnn_o_ft1' tag=cifar_s3 \
    tstart=20 tend=30021 tskip=1500 \
    fine_tune=16 data.bs=8 train.epochs=1 data.path='./data/cifar/scenario3_markov4.pkl'

python3 02_train.py  net.type=cnn_cifar name='cnn_o_bgd' tag=cifar_s3 \
    tstart=20 tend=30021 tskip=1500 bgd=True \
    fine_tune=16 data.bs=8 train.epochs=1 data.path='./data/cifar/scenario3_markov4.pkl'

python3 02_train.py name='erm_cnn' tag=cifar_s3_markov2 \
    tstart=20 tend=30021 tskip=1500 \
    net.type=cnn_cifar \
    train.epochs=100 data.bs=32 data.path='./data/cifar/scenario3_markov2.pkl'

python3 02_train.py name='prospective_cnn_o' tag=cifar_s3_markov2 \
    tstart=20 tend=30021 tskip=1500 \
    net.type=prospective_cnn_cifar net.time_last=True \
    train.epochs=100 data.bs=32 data.path='./data/cifar/scenario3_markov2.pkl'

