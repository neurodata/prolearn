# #### MNIST ####

python run_proformer.py -m device='cuda:0' method="proformer" multihop=True encoding="freq" epochs=150 batchsize=32 t=0,200,500,700,1000,1200,1500,1700,2000,2500 hydra.launcher.n_jobs=6
python run_proformer.py -m device='cuda:1' method="proformer" multihop=True encoding="vanilla" epochs=150 batchsize=32 t=0,200,500,700,1000,1200,1500,1700,2000,2500 hydra.launcher.n_jobs=6

python run_proformer.py -m device='cuda:2' method="proformer" multihop=False encoding="freq" epochs=150 batchsize=32 t=0,200,500,700,1000,1200,1500,1700,2000,2500 hydra.launcher.n_jobs=6
python run_proformer.py -m device='cuda:3' method="proformer" multihop=False encoding="vanilla" epochs=150 batchsize=32 t=0,200,500,700,1000,1200,1500,1700,2000,2500 hydra.launcher.n_jobs=6


# #### CIFAR-10 ####

# python run_vision_multi.py -m device='cuda:1' method="conv_proformer" multihop=True epochs=300 batchsize=16 t=0,200,500,700,1000,1200,1500,1700,2000,2500,3000,4000 hydra.launcher.n_jobs=1
# python run_vision_multi.py -m device='cuda:2' method="conv_proformer" multihop=False epochs=300 batchsize=16 t=0,200,500,700,1000,1200,1500,1700,2000,2500,3000,4000 hydra.launcher.n_jobs=1
