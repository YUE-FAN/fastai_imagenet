from fastai.vision import *
from fastai.vision.models import resnet50
from resnetfy import Resnet50
from fastai.distributed import *
from fastai.callbacks import SaveModelCallback
from vggfy import VGG16

from fastai.distributed import *
import argparse
torch.backends.cudnn.benchmark = True
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int)
args = parser.parse_args()
torch.cuda.set_device(args.local_rank)
torch.distributed.init_process_group(backend='nccl', init_method='env://')


class my_CSVLogger(LearnerCallback):
    def __init__(self, learn:Learner, filename: str = 'history'):
        super().__init__(learn)
        self.filename,self.path = filename,self.learn.path/f'{filename}.csv'
        self.names = ['epoch', 'train_loss', 'valid_loss', 'accuracy', 'top_k_accuracy', 'lr']

    def read_logged_file(self):
        "Read the content of saved file"
        return pd.read_csv(self.path)

    def on_train_begin(self, **kwargs: Any) -> None:
        "Prepare file with metric names."
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open('w')
        self.file.write(','.join(self.names) + '\n')

    def on_epoch_end(self, epoch: int, smooth_loss: Tensor, last_metrics: MetricsList, **kwargs: Any) -> bool:
        "Add a line with `epoch` number, `smooth_loss` and `last_metrics`."
        last_metrics = ifnone(last_metrics, [])
        stats = [str(stat) if isinstance(stat, int) else f'{stat:.6f}'
                 for name, stat in zip(self.names, [epoch, smooth_loss] + last_metrics)]
        stats.append(str(self.learn.recorder.lrs[-1]))
        print(stats)
        str_stats = ','.join(stats)
        self.file.write(str_stats + '\n')

    def on_train_end(self, **kwargs: Any) -> None:
        "Close the file."
        self.file.close()



print(num_cpus())
print("sadsadsad")
# path = untar_data(URLs.CIFAR, dest="./data/")
path = Path('/fan/datasets/imagenet/')
ds_tfms = ([*rand_resize_crop(224), flip_lr(p=0.5)], [])
# ds_tfms = None
# n_gpus = 4
data = ImageDataBunch.from_folder(path, valid='val', ds_tfms=ds_tfms, bs=256//8, num_workers=8, size=224, resize_method=ResizeMethod.CROP).normalize(imagenet_stats)

# learn = Learner(data, resnet50(), metrics=accuracy)
optm = partial(optim.SGD, momentum=0.9)
learn = Learner(data, Resnet50(0, 1000, True, 99), opt_func=optm, metrics=[accuracy, top_k_accuracy]).distributed(args.local_rank)
learn.to_fp32()
# learn.model = nn.parallel.DistributedDataParallel(learn.model)
# learn.model = nn.DataParallel(learn.model)
# print(learn.summary())



print('start training...')
csver = my_CSVLogger(learn, filename='log')
best_saver = SaveModelCallback(learn, every='improvement', monitor='accuracy', name='Resnet50_best')
checkpoint = SaveModelCallback(learn, every='epoch', name='checkpoint')
learn.fit_one_cycle(100, 0.3, wd=0.4, callbacks=[best_saver, checkpoint, csver])


# data = ImageDataBunch.from_folder(path, valid='test', ds_tfms=(tfms, []), bs=512).normalize(cifar_stats)
# ds = data.train_ds
# learn = Learner(data, resnet50(), metrics=accuracy).to_fp16()
# learn.fit_one_cycle(30, 3e-3, wd=0.4, div_factor=10, pct_start=0.5)
