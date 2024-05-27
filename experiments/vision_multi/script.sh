# #### MNIST ####

# lower bound baseline (oracle learner)
python run_baseline_1.py -m device='cuda:0' method="cnn" epochs=200 batchsize=64 t=0,200,500,700,1000,1200,1500,1700,2000,2500,3000,4000 hydra.launcher.n_jobs=10

# upper bound baseline (OGD)
python run_baseline_2.py -m device='cuda:1' method="cnn" epochs=100 batchsize=64 hydra.launcher.n_jobs=1

# cnn
python run_vision_multi.py -m device='cuda:0' method="cnn" epochs=500 batchsize=64 t=0,200,500,700,1000,1200,1500,1700,2000,2500,3000,4000 hydra.launcher.n_jobs=8

# timecnn
python run_vision_multi.py -m device='cuda:1' method="timecnn" epochs=500 batchsize=64 t=0,200,500,700,1000,1200,1500,1700,2000,2500,3000,4000 hydra.launcher.n_jobs=8

# prosp. transformer
python run_vision_multi.py -m device='cuda:2' method="proformer" multihop=True epochs=300 batchsize=32 t=0,200,500,700,1000,1200,1500,1700,2000,2500,3000,4000 hydra.launcher.n_jobs=6

# transformer
python run_vision_multi.py -m device='cuda:3' method="proformer" multihop=False epochs=300 batchsize=32 t=0,200,500,700,1000,1200,1500,1700,2000,2500,3000,4000 hydra.launcher.n_jobs=6


# #### CIFAR-10 ####

# upper bound baseline (Oracle)
python run_baseline_1.py -m device='cuda:0' method="resnet" epochs=1000 batchsize=64 t=0,200,500,700,1000,1200,1500,1700,2000,2500,3000,4000 hydra.launcher.n_jobs=8
python run_baseline_1.py -m device='cuda:2' method="resnet" epochs=1000 batchsize=64 t=4000 hydra.launcher.n_jobs=1

# upper bound baseline (OGD)
python run_baseline_2.py -m device='cuda:1' method="resnet" epochs=500 batchsize=64 hydra.launcher.n_jobs=1

# resnet
python run_vision_multi.py -m device='cuda:0' method="resnet" epochs=1000 batchsize=64 t=0,200,500,700,1000,1200,1500,1700,2000,2500,3000,4000 hydra.launcher.n_jobs=8

# time-resnet
python run_vision_multi.py -m device='cuda:1' method="timeresnet" epochs=1000 batchsize=64 t=0,200,500,700,1000,1200,1500,1700,2000,2500,3000,4000 hydra.launcher.n_jobs=8

# prosp. transformer
python run_vision_multi.py -m device='cuda:1' method="conv_proformer" multihop=True epochs=300 batchsize=16 t=0,200,500,700,1000,1200,1500,1700,2000,2500,3000,4000 hydra.launcher.n_jobs=1

# transformer
python run_vision_multi.py -m device='cuda:2' method="conv_proformer" multihop=False epochs=300 batchsize=16 t=0,200,500,700,1000,1200,1500,1700,2000,2500,3000,4000 hydra.launcher.n_jobs=1


# # lower bound baseline
# python run_baseline.py -m t=0,100,200,500,700,1000,1200,1500,1700,2000,2500,3000,4000 method="resnet" device='cuda:1' hydra.launcher.n_jobs=10

# # for others
# python run_vision_multi.py -m t=0,100,200,500,700,1000,1200,1500,1700,2000,2500,3000,4000 method="resnet" device='cuda:0' hydra.launcher.n_jobs=10
# python run_vision_multi.py -m t=0,100,200,500,700,1000,1200,1500,1700,2000,2500,3000,4000 method="timeresnet" device='cuda:1' hydra.launcher.n_jobs=10

# # for conv_proformer
# python run_vision_multi.py -m t=0,100,200,500,700,1000,1200,1500,1700,2000,2500,3000,4000 method="conv_proformer" device='cuda:0' hydra.launcher.n_jobs=4

# CUDA_VISIBLE_DEVICES=1 python run_vision_multi.py -m t=1000 method="conv_proformer" device='cuda:0' hydra.launcher.n_jobs=1