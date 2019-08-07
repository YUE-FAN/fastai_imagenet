# fastai_imagenet

python -m torch.distributed.launch --nproc_per_node={num_gpus} {script_name}

python cifar_distri.py -a resnet50 --layer 99 --dataset cifar10 --depth 110 --epochs 5 --schedule 2 3 --gamma 0.1 --wd 1e-4 --checkpoint checkpoints/del/dell --multiprocessing-distributed  --dist-url tcp://127.0.0.1:8888

python cifar.py -a vgg16 --dataset cifar10 --depth 110 --epochs 5 --schedule 2 3 --gamma 0.1 --wd 1e-4 --checkpoint checkpoints/del/dell --gpu-id 0

python main_torch.py -a resnet50 --layer 99 --dataset /datasets/imagenet/ --epochs 100 --schedule 30 60 --gamma 0.1 --wd 1e-4 --checkpoint /trained-models/imagenet/resnet50_torch/ --multiprocessing-distributed  --dist-url tcp://127.0.0.1:8888 --ngpus_per_node 8 --lr 0.6 --workers 32

python main_torch.py -a resnet50_1x1 --layer 35 --dataset /BS/database11/ILSVRC2012/ --epochs 90 --schedule 30 60 --train-batch 256 --checkpoint /BS/yfan/work/trained-models/dconv/checkpoints/imagenet/resnet501x1_90_lr0.1_bs256/resnet501x1_3542_90 --multiprocessing-distributed --ngpus_per_node 3 --workers 32

## To Yongqin:
Requirement:
Python 3.6.7
numpy 1.16.2
scipy 1.2.1
Pillow 5.4.1
torch 1.0.0 and corresponding torchvision

You may create a conda env and run a tmux session. Inside the session, just "bash vgg161d.sh" or "bash d1_resnet50.sh"

Inside the .sh files, each line trains one independent model. You can divide those lines into several .sh files so that they can be run in parallel. Please remember to specify the --dataset to the location of imagnet dataset and --checkpoint to the location where you would like to store the model (the folder will be create it automatically).


![equation]$$P(W)^c$$
