###### Synthetic - Scenario 2
python3 02_train.py net.type=mlp name='mlp_erm' tag=scenario2_v2  \
    tstart=20 tskip=200 tend=2001

python3 02_train.py net.type=prospective_mlp name='mlp_prospective' tag=scenario2_v2 \
    tstart=20 tskip=200 tend=2001 

python3 02_train.py net.type=mlp name='mlp_ft1' tag=scenario2_v2 \
    tstart=20 tskip=200 tend=2001  \
    train.epochs=1 fine_tune=16 data.bs=8

python3 02_train.py net.type=mlp name='mlp_bgd' tag=scenario2_v2 \
    tstart=20 tskip=200 tend=2001 \
    train.epochs=1 fine_tune=16 data.bs=8 bgd=True

######### MNIST - Scenario 2
python3 02_train.py  net.type=mlp_mnist name='erm_mlp' tag=mnist_s2_v2 \
    tstart=20 tend=5021 tskip=250 \
    train.epochs=10 data.bs=32 data.path='./data/mnist/scenario2.pkl'

python3 02_train.py  net.type=prospective_mlp_mnist name='prospective_mlp' tag=mnist_s2_v2  \
    tstart=20 tend=5021 tskip=250 \
    train.epochs=100 data.bs=32 data.path='./data/mnist/scenario2.pkl'

python3 02_train.py net.type=mlp_mnist name='mlp_ft1' tag=mnist_s2_v2 \
    tstart=20 tend=5001 tskip=250 \
    train.epochs=1 fine_tune=16 data.bs=8 data.path='./data/mnist/scenario2.pkl'
 
 
python3 02_train.py net.type=mlp_mnist name='mlp_bgd' tag=mnist_s2_v2 \
    tstart=20 tend=5001 tskip=250 \
    train.epochs=1 fine_tune=16 data.bs=8 bgd=True data.path='./data/mnist/scenario2.pkl'

######### CIFAR - Scenario 2
python3 02_train.py  net.type=cnn_cifar name='erm_cnn' tag=cifar_s2 \
    tstart=20 tend=30021 tskip=1500 \
    train.epochs=100 data.bs=32 data.path='./data/cifar/scenario2.pkl'

python3 02_train.py  net.type=prospective_cnn_cifar name='prospective_cnn_o' tag=cifar_s2 \
    tstart=20 tend=30021 tskip=1500 \
    train.epochs=100 data.bs=32 data.path='./data/cifar/scenario2.pkl'

python3 02_train.py  net.type=cnn_cifar name='cnn_o_ft1' tag=cifar_s2 \
    tstart=20 tend=30021 tskip=500 \
    fine_tune=16 data.bs=8 train.epochs=1 data.path='./data/cifar/scenario2.pkl'

python3 02_train.py  net.type=cnn_cifar name='cnn_o_bgd' tag=cifar_s2 \
    tstart=20 tend=30021 tskip=500 bgd=True \
    fine_tune=16 data.bs=8 train.epochs=1 data.path='./data/cifar/scenario2.pkl'
