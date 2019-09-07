import torch
from torch.autograd import Variable
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms


class ModelOutputs:
    # TODO: this is now only suitable for ResNet
    """ Class for making a forward pass, and getting:
    1. The network output logits x.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, target_layers, use_cuda):
        self.model = model
        self.target_layers = target_layers  # ['identity_block_4_2'] for resnet50
        self.gradients = []
        self.use_cuda = use_cuda

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def get_gradients(self):
        return self.gradients

    def __call__(self, x):
        target_feature_map = []
        if self.use_cuda:
            m = self.model.module
        else:
            m = self.model
        for name, module in m._modules.items():
            if name in self.target_layers:
                return_intermediate_activations = True
                x, intermediate_activations = module(x, return_intermediate_activations)
                intermediate_activations.register_hook(self.save_gradient)
                target_feature_map += [intermediate_activations]
            elif name == 'avgpool':
                x = module(x)
                x = x.view(x.size(0), -1)
            else:
                x = module(x)
        return target_feature_map, x


class GradCam:
    def __init__(self, model, target_layer_names, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, target_layer_names, use_cuda)

    def forward(self, input):
        return self.model(input)

    def __call__(self, inputs, cam_indexes):
        """
        :param inputs: torch.tensor, [bs, C, H, W]
        :param cam_indexes:  torch.tensor, [bs,]
        :return: logits in [bs, num_class] (torch.tensor) and cams in [bs, h, w] (numpy.array)
        """
        if self.cuda:
            features, logits = self.extractor(inputs.cuda())
        else:
            features, logits = self.extractor(inputs)

        one_hot = np.zeros(logits.size(), dtype=np.float32)  # [bs, num_class]
        one_hot[np.arange(one_hot.shape[0]), cam_indexes] = 1
        one_hot = Variable(torch.from_numpy(one_hot), requires_grad=True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * logits, dim=1)
        else:
            one_hot = torch.sum(one_hot * logits, dim=1)

        self.model.zero_grad()
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        # grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()  # [bs, c, h, w]
        grads_val = self.extractor.get_gradients()[-1].detach().data  # [bs, c, h, w]
        targets = features[-1]  # [bs, c, h, w]
        # targets = targets.cpu().data.numpy()
        targets = targets.detach().data
        # weights = np.mean(grads_val, axis=(2, 3))  # [bs, c]
        weights = torch.mean(grads_val, dim=(2, 3), keepdim=True)  # [bs, c]

        cams = weights * targets  # [bs, c, h, w]
        cams = cams.sum(dim=1).cpu().numpy()  # [bs, h, w]

        return logits, cams


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def draw_imagenetgrad_cam(model, img_paths, class_idxs, target_layer_names, use_cuda):
    """
    Given images and class indexes, return grad-camed images (to show or save)
    :param model: a loaded model
    :param img_paths: a list of str
    :param class_idxs: a list of indexes
    :return: grad_camed_imgs
    """
    assert len(img_paths) == len(class_idxs), 'the number of images is not equal to the number of CAM indexes!'
    grad_cam = GradCam(model, target_layer_names, use_cuda=use_cuda)

    # load images
    inputs = torch.empty(0).float()
    tfms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    for path in img_paths:
        img = pil_loader(path)
        sample = tfms(img).unsqueeze(0)
        inputs = torch.cat((inputs, sample), dim=0)
    if use_cuda:
        inputs = inputs.cuda()
    class_idxs = torch.tensor(class_idxs)

    # compute CAMs
    _, cams = grad_cam(inputs, class_idxs)

    # draw CAMs on images
    grad_camed_imgs = []
    for i, cam in enumerate(cams):
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)

        img = cv2.imread(img_paths[i], 1)
        img = np.float32(cv2.resize(img, (224, 224))) / 255
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        grad_camed_imgs.append(np.uint8(255 * cam))
    return grad_camed_imgs

