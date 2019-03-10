# fastai_imagenet

python -m torch.distributed.launch --nproc_per_node={num_gpus} {script_name}

python cifar_distri.py -a resnet50 --layer 99 --dataset cifar10 --depth 110 --epochs 5 --schedule 2 3 --gamma 0.1 --wd 1e-4 --checkpoint checkpoints/del/dell --multiprocessing-distributed  --dist-url tcp://127.0.0.1:8888

python cifar.py -a vgg16 --dataset cifar10 --depth 110 --epochs 5 --schedule 2 3 --gamma 0.1 --wd 1e-4 --checkpoint checkpoints/del/dell --gpu-id 0

python main_torch.py -a resnet50 --layer 99 --dataset /datasets/imagenet/ --epochs 100 --schedule 30 60 --gamma 0.1 --wd 1e-4 --checkpoint /trained_models/imagenet/resnet50_torch/ --multiprocessing-distributed  --dist-url tcp://127.0.0.1:8888 --ngpus_per_node 8 --lr 0.1 --workers 32
