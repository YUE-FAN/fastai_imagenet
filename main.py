from fastai.vision import *
from fastai.vision.models import resnet50
from resnetfy import Resnet50
from fastai.distributed import *

from vggfy import VGG16

from fastai.distributed import *
import argparse
torch.backends.cudnn.benchmark = True
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int)
args = parser.parse_args()
torch.cuda.set_device(args.local_rank)
torch.distributed.init_process_group(backend='nccl', init_method='env://')

def center_crop(size:int):
    return [crop(size=size)]




print(num_cpus())
print("sadsadsad")
path = untar_data(URLs.CIFAR, dest="./data/")
# tfms = [rand_resize_crop(224), flip_lr(p=0.5)]
ds_tfms = ([*rand_pad(4, 32), flip_lr(p=0.5)], [*center_crop(32)])
# ds_tfms = None
# n_gpus = 4
data = ImageDataBunch.from_folder(path, valid='test', ds_tfms=ds_tfms,  bs=512, num_workers=6).normalize(cifar_stats)

# learn = Learner(data, resnet50(), metrics=accuracy)

learn = Learner(data, Resnet50(0, 10, True, 99), metrics=[accuracy, top_k_accuracy]).distributed(args.local_rank)
learn.to_fp16()
# learn.model = nn.parallel.DistributedDataParallel(learn.model)
# learn.model = nn.DataParallel(learn.model)
# print(learn.summary())



print('start training...')
learn.fit_one_cycle(35, 3e-3, wd=0.4)


# data = ImageDataBunch.from_folder(path, valid='test', ds_tfms=(tfms, []), bs=512).normalize(cifar_stats)
# ds = data.train_ds
# learn = Learner(data, resnet50(), metrics=accuracy).to_fp16()
# learn.fit_one_cycle(30, 3e-3, wd=0.4, div_factor=10, pct_start=0.5)