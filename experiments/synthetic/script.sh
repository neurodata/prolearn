# Baseline 1 (Oracle)
python run_baseline_1.py -m device='cuda:0' method="mlp" epochs=300 batchsize=64 t=0,100,200,500,700,1000,1200,1500,1700,2000,2500 hydra.launcher.n_jobs=8

# MLP
python run_synthetic.py -m device='cuda:0' method="mlp" epochs=300 batchsize=64 t=0,100,200,500,700,1000,1200,1500,1700,2000,2500 hydra.launcher.n_jobs=8

# time MLP
# python run_synthetic.py -m device='cuda:1' method="timemlp" epochs=300 batchsize=64 t=0,100,200,500,700,1000,1200,1500,1700,2000,2500 hydra.launcher.n_jobs=8

# prosp. transformer
# python run_synthetic.py -m device='cuda:2' method="proformer" multihop=True epochs=300 batchsize=64 t=0,100,200,500,700,1000,1200,1500,1700,2000,2500 hydra.launcher.n_jobs=8

# transformer
# python run_synthetic.py -m device='cuda:3' method="proformer" multihop=False epochs=300 batchsize=64 t=0,100,200,500,700,1000,1200,1500,1700,2000,2500 hydra.launcher.n_jobs=8

