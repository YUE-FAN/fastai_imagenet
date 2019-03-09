from fastai.vision import *
from fastai.vision.models import resnet50
from resnetfy import Resnet50
from fastai.distributed import *
from fastai.callbacks import SaveModelCallback, ReduceLROnPlateauCallback, TrackerCallback
from vggfy import VGG16

from fastai.distributed import *
import argparse
torch.backends.cudnn.benchmark = True
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int)
args = parser.parse_args()
torch.cuda.set_device(args.local_rank)
torch.distributed.init_process_group(backend='nccl', init_method='env://')


class ReduceLR(TrackerCallback):
    "A `TrackerCallback` that reduces learning rate when a metric has stopped improving."
    def __init__(self, learn:Learner, monitor:str='val_loss', mode:str='auto', patience:int=0, factor:float=0.2,
                 min_delta:int=0):
        super().__init__(learn, monitor=monitor, mode=mode)
        self.patience,self.factor,self.min_delta = patience,factor,min_delta
        if self.operator == np.less:  self.min_delta *= -1

    def on_train_begin(self, **kwargs:Any)->None:
        "Initialize inner arguments."
        self.wait, self.opt = 0, self.learn.opt
        super().on_train_begin(**kwargs)

    def on_epoch_end(self, epoch, **kwargs:Any)->None:
        if epoch == 30 or epoch == 60:
            self.opt.lr *= self.factor
            print(f'Epoch {epoch}: reducing lr to {self.opt.lr}')


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
path = Path('/datasets/imagenet/')
ds_tfms = ([*rand_resize_crop(224), brightness(change=(0.4,0.6)), contrast(scale=(0.7,1.3)), flip_lr(p=0.5)], [])
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
# lr_scheduler = ReduceLROnPlateauCallback(learn, patience=5, factor=0.1, monitor='accuracy', min_delta=0)
lr_scheduler = ReduceLR(learn, patience=5, factor=0.1, monitor='accuracy', min_delta=0)
csver = my_CSVLogger(learn, filename='/logfiles/log')
best_saver = SaveModelCallback(learn, every='improvement', monitor='accuracy', name='/trained-models/Resnet50_best')
checkpoint = SaveModelCallback(learn, every='epoch', name='/trained-models/checkpoint')
learn.fit(120, 0.1, wd=1e-4, callbacks=[best_saver, checkpoint, csver, lr_scheduler])
# learn.fit_one_cycle(100, 0.3, wd=0.4, callbacks=[best_saver, checkpoint, csver])


# data = ImageDataBunch.from_folder(path, valid='test', ds_tfms=(tfms, []), bs=512).normalize(cifar_stats)
# ds = data.train_ds
# learn = Learner(data, resnet50(), metrics=accuracy).to_fp16()
# learn.fit_one_cycle(30, 3e-3, wd=0.4, div_factor=10, pct_start=0.5)
