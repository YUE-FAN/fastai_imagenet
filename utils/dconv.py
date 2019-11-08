# This code come from https://github.com/oeway/pytorch-deform-conv
# Thank him, and I change some code.

from torch.autograd import Variable
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import random
from scipy.ndimage.interpolation import map_coordinates as sp_map_coordinates

cuda_number = 0


class Dconv_cos(nn.Module):
    """
    re-arrange the spatial layout of the activations.
    Then perform a dilated convolution to achieve the dconv.
    We don't have the concept of the OFFSET at all. Instead, we generate
    the new sampling locations directly. Even though I believe offset is
    easier to obtain in terms of the learning, but our deformation has
    nothing to do with the learning, it is a parameter-free method.

    The key problems in our method are:
    1. How to obtain the new sampling locations.
        For now we choose 9 locations where the activations are most similar
        to the center one. And we discard the spatial distance between them.
        TODO: consider the spatial distance and use other similarity measure
    2. How to arrange a new squared layout for the upcoming convolution.
        For now we just go through all the activations from the left to the
        right and then top down.
        TODO: come up with a more descent arrangement
    """

    def __init__(self, height, width, inplane, outplane, win_range=100, kernel_size=3):
        """

        :param win_range: defines the size of the search window (for now it has to be odd)
        :param height: the height of the input feature map
        :param width: the width of the input feature map
        """
        super(Dconv_cos, self).__init__()
        self.h = height
        self.w = width
        self.win_range = win_range

        # offset dict is a dict of the same length of the number of activations in the input feature map.
        # Each key is a tuple denoting the location of the activation, the value is a list of the
        # indices of its neighbors, the length of the list is win_range * win_range.
        self.offset_dict = {}
        self.spatial_dist_dict = {}
        min_len = 10000  # the smallest length of the idx_list
        for key_i in range(self.h):
            for key_j in range(self.w):

                idx_list = []  # the list for the indices of the neighbors of the point (key_i, key_j)
                dist_list = []  # the list for the euc dist between the neighbors and the point (key_i, key_j)

                # go through all the positions nearby and add them to the list
                for i in range(self.h):
                    for j in range(self.w):
                        if np.abs(i - key_i) <= (self.win_range - 1) / 2 and np.abs(j - key_j) <= (
                                self.win_range - 1) / 2:
                            idx_list.append(self.loc_1D([i, j]))
                            dist_list.append(np.linalg.norm([i - key_i, j - key_j]))  # TODO: try hamming distance
                if len(idx_list) < min_len:
                    min_len = len(idx_list)

                self.offset_dict[(key_i, key_j)] = torch.tensor(idx_list).cuda(cuda_number)
                self.spatial_dist_dict[(key_i, key_j)] = torch.tensor(dist_list).cuda(cuda_number)

        if min_len < kernel_size * kernel_size:
            raise Exception('the window size is too small to contain the kernel!')
        # since each location has its own offset, the re-arranged feature map should have a larger size than the x,
        # namely, x is 32x32 and kernel is 3x3, then x_offset should be 96x96. But only 32x32 locations out of 96x96
        # are needed for convolution. Thus a dilated convolution should be used. The stride should be the same as the
        # size of the kernel and the padding should be disabled. For now, only 3x3 convolution is supported.
        # TODO: support convolutions with any kernel size
        self.kernel_size = kernel_size
        self.dilated_conv = nn.Conv2d(inplane, outplane, kernel_size=self.kernel_size, stride=self.kernel_size,
                                      padding=0, bias=False)

        # the relative locations of an activation, those locations has to be filled, it is the same size of the kernel
        self.directions = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 0], [0, 1], [1, -1], [1, 0], [1, 1]]

    def forward(self, x):
        """
        Enumerate all (key, idx_list) pairs in the offset_dict.
        For each pair, first compute distance list, and then re-arrange x accordingly and assign x to the x_offset.
        Finally a dilated convolution performed on the x_offset.
        """
        x_shape = x.size()  # [128, 3, 32, 32]
        x = x.view(x_shape[0], x_shape[1], x_shape[2] * x_shape[3])  # [128, 3, 32*32]

        # assign the first n most similar activations from the neighbors to the squared neighbourhood of "key"
        x_offset = torch.empty(x_shape[0], x_shape[1], x_shape[2] * self.kernel_size,
                               x_shape[3] * self.kernel_size).cuda(cuda_number)
        # plot_data = []
        # dist_lists = np.empty((64, 100, 64))
        # cc = 0
        for key, idx_list in self.offset_dict.items():
            tmpX1 = x[:, :, self.loc_1D(key)].view(x_shape[0], x_shape[1], 1)  # [128, 3, 1]
            tmpX2 = x[:, :, idx_list].view(x_shape[0], x_shape[1], len(idx_list))  # [128, 3, 7*7]

            # distance_list contains the cosine similarity between key and every location in idx_list.
            # It is batchified.
            dist_list = F.cosine_similarity(tmpX1.permute(0, 2, 1), tmpX2.permute(0, 2, 1), 2, 1e-6)  # [128, 7*7]
            # print(dist_list.detach().cpu().numpy())
            # print(type(dist_list.detach().cpu().numpy()))
            # dist_lists[cc, :, :] = dist_list.detach().cpu().numpy()
            # cc += 1
            # sort in an descending way because it is a similarity measure
            delll, orders = torch.sort(dist_list, dim=1, descending=True)  # TODO: the correct one is descending=True
            # sort idx_list according to dist_list
            sorted_idx_list = torch.take(idx_list, orders)  # [128, 7*7]
            # take the first several smallest ones
            sorted_idx_list = sorted_idx_list[:, 0: self.kernel_size * self.kernel_size]  # [128, 3*3]
            # now sort the idx_list ascendingly
            sorted_idx_list, _ = torch.sort(sorted_idx_list, dim=1, descending=False)
            # the following two lines show the deformed 1D sampling locations, I'd like to draw an interactive graph
            # based on it, where the sampling locations will light up when you click on an activation
            # print(key)
            # print(sorted_idx_list)
            # plot_data.append(sorted_idx_list.cpu().numpy())

            for i, relativ_loc in enumerate(self.directions):
                indd = sorted_idx_list[:, i]
                # print(indd)
                # print(x[torch.arange(0, 128, 1).long(), :, indd].size())
                x_offset[:, :, (key[0] * self.kernel_size + 1) + relativ_loc[0],
                (key[1] * self.kernel_size + 1) + relativ_loc[1]] = x[torch.arange(0, x_shape[0], 1).long(), :, indd]

        # np.save('/nethome/yuefan/fanyue/dconv/layout.npy', plot_data)
        # np.save('/nethome/yuefan/fanyue/dconv/dist_lists.npy', dist_lists)
        # apply dilated convolution so that skip the undefined locations
        x_offset = self.dilated_conv(x_offset)

        return x_offset

    def loc_2D(self, loc_1D):
        # turn a 1D coordinate into a 2D coordinate
        return [loc_1D // self.w, loc_1D % self.w]

    def loc_1D(self, loc_2D):
        # turn a 2D coordinate into a 1D coordinate
        return loc_2D[0] * self.w + loc_2D[1]


class Dconv_drop(nn.Module):
    """
    Deformable convolution with random sampling locations.
    The sampling locations are generated at the very beginning and fixed during the training.
    The fixed random sampling locations are shared over all kernels, but only vary over spatial location
    on the feature map.
    """

    def __init__(self, height, width, inplane, outplane, win_range=100, kernel_size=3):
        """
        :param win_range: defines the size of the search window (for now it has to be odd)
        :param height: the height of the input feature map
        :param width: the width of the input feature map
        """
        super(Dconv_drop, self).__init__()
        self.h = height
        self.w = width
        self.win_range = win_range

        # offset dict is a dict of the same length of the number of activations in the input feature map.
        # Each key is a tuple denoting the location of the activation, the value is a list of the
        # indices of its neighbors, the length of the list is win_range * win_range.
        self.offset_dict = {}
        # sample_dict is a dict of the same format as offset dict. For each key, the value is a list of length 9
        # containing the fixed random sampling locations for that activation.
        self.sample_dict = {}
        min_len = 10000  # the smallest length of the idx_list
        for key_i in range(self.h):
            for key_j in range(self.w):

                idx_list = []  # the list for the indices of the neighbors of the point (key_i, key_j)

                # go through all the positions nearby and add them to the list
                for i in range(self.h):
                    for j in range(self.w):
                        if np.abs(i - key_i) <= (self.win_range - 1) / 2 and np.abs(j - key_j) <= (
                                self.win_range - 1) / 2:
                            idx_list.append(self.loc_1D([i, j]))
                if len(idx_list) < min_len:
                    min_len = len(idx_list)
                    if min_len < kernel_size * kernel_size:
                        raise Exception('the window size is too small to contain the kernel!')
                self.offset_dict[(key_i, key_j)] = torch.tensor(idx_list).cuda(cuda_number)
                # randomly select 9 sampling locations within the window
                self.sample_dict[(key_i, key_j)] = torch.tensor(
                    random.sample(idx_list, kernel_size * kernel_size)).cuda(cuda_number)

        # since each location has its own offset, the re-arranged feature map should have a larger size than the x,
        # namely, x is 32x32 and kernel is 3x3, then x_offset should be 96x96. But only 32x32 locations out of 96x96
        # are needed for convolution. Thus a dilated convolution should be used. The stride should be the same as the
        # size of the kernel and the padding should be disabled. For now, only 3x3 convolution is supported.
        # TODO: support convolutions with any kernel size
        self.kernel_size = kernel_size
        self.dilated_conv = nn.Conv2d(inplane, outplane, kernel_size=self.kernel_size, stride=self.kernel_size,
                                      padding=0, bias=False)

        # the relative locations of an activation, those locations has to be filled, it is the same size of the kernel
        self.directions = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 0], [0, 1], [1, -1], [1, 0], [1, 1]]

    def forward(self, x):
        x_shape = x.size()  # [128, 3, 32, 32]
        x = x.view(x_shape[0], x_shape[1], x_shape[2] * x_shape[3])  # [128, 3, 32*32]

        x_offset = torch.empty(x_shape[0], x_shape[1], x_shape[2] * self.kernel_size,
                               x_shape[3] * self.kernel_size).cuda(cuda_number)

        for key, idx_list in self.sample_dict.items():
            for i, relativ_loc in enumerate(self.directions):
                x_offset[:, :, (key[0] * self.kernel_size + 1) + relativ_loc[0],
                (key[1] * self.kernel_size + 1) + relativ_loc[1]] = x[:, :, idx_list[i]]

        x_offset = self.dilated_conv(x_offset)
        return x_offset

    def loc_2D(self, loc_1D):
        # turn a 1D coordinate into a 2D coordinate
        return [loc_1D // self.w, loc_1D % self.w]

    def loc_1D(self, loc_2D):
        # turn a 2D coordinate into a 1D coordinate
        return loc_2D[0] * self.w + loc_2D[1]


class Dconv_rand(nn.Module):
    """
    Deformable convolution with random sampling locations.
    Channles are not shuffled, they follow the movement of the spatial locations.
    The sampling locations are generated for each forward pass during the training.
    """

    def __init__(self, inplane, outplane, kernel_size, stride, padding):
        super(Dconv_rand, self).__init__()
        print('cifar Dconv_rand is used')
        self.dilated_conv = nn.Conv2d(inplane, outplane, kernel_size=kernel_size, stride=stride, padding=padding,
                                      bias=False)

    def forward(self, x):
        x_shape = x.size()  # [128, 3, 32, 32]
        x = x.view(x_shape[0], x_shape[1], x_shape[2] * x_shape[3])  # [128, 3, 32*32]
        x_offset = torch.empty(x_shape[0], x_shape[1], x_shape[2] * x_shape[3]).cuda(cuda_number)
        # np.save('/nethome/yuefan/fanyue/dconv/x.npy', x.detach().cpu().numpy())
        perm = torch.randperm(x_shape[2] * x_shape[3])
        x_offset[:, :, :] = x[:, :, perm]

        x_offset = x_offset.view(x_shape[0], x_shape[1], x_shape[2], x_shape[3])

        return self.dilated_conv(x_offset)


class Dconv_crand(nn.Module):
    """
    Deformable convolution with random sampling locations.
    This is the channel version of rand above.
    The sampling locations are generated for each forward pass during the training.
    """

    def __init__(self, inplane, outplane, kernel_size, stride, padding):
        super(Dconv_crand, self).__init__()
        print('cifar Dconv_crand is used')
        self.dilated_conv = nn.Conv2d(inplane, outplane, kernel_size=kernel_size, stride=stride, padding=padding,
                                      bias=False)

    def forward(self, x):
        x_shape = x.size()  # [128, 3, 32, 32]
        # x_offset = torch.empty(x_shape[0], x_shape[1], x_shape[2], x_shape[3]).cuda(cuda_number)
        perm = torch.randperm(x_shape[1])
        x = x[:, perm, :, :]
        return self.dilated_conv(x)


class Dconv_shuffle(nn.Module):
    """
    Deformable convolution with random shuffling of the feature map.
    Random shuffling only happened within each page independently.
    The sampling locations are generated for each forward pass during the training.
    """

    def __init__(self, inplane, outplane, kernel_size, stride, padding):
        super(Dconv_shuffle, self).__init__()
        print('cifar Dconv_shuffle is used')
        self.dilated_conv = nn.Conv2d(inplane, outplane, kernel_size=kernel_size, stride=stride, padding=padding,
                                      bias=False)
        self.indices = None

    def _setup(self, inplane, spatial_size):
        self.indices = np.empty((inplane, spatial_size), dtype=np.int64)
        for i in range(inplane):
            self.indices[i, :] = np.arange(self.indices.shape[1]) + i * self.indices.shape[1]

    def forward(self, x):
        # x_shape = x.size()  # [128, 3, 32, 32]
        # x = x.view(x_shape[0] * x_shape[1] * x_shape[2] * x_shape[3], )  # [128, 3*32*32]
        # x_offset = torch.empty(x_shape[0], x_shape[1], x_shape[2], x_shape[3]).cuda(cuda_number)
        # # np.save('/nethome/yuefan/fanyue/dconv/x.npy', x.detach().cpu().numpy())
        # permth = torch.empty((x_shape[0] * x_shape[1] * x_shape[2] * x_shape[3], )).float()
        # for ii in range(x_shape[0]):
        #     perm = torch.empty((x_shape[1] * x_shape[2] * x_shape[3], )).float()
        #     for i in range(x_shape[1]):
        #         a = torch.randperm(x_shape[2] * x_shape[3]) + i * x_shape[2] * x_shape[3]
        #         perm[i * x_shape[2] * x_shape[3]:(i+1) * x_shape[2] * x_shape[3]] = a
        #     permth[ii * x_shape[1] * x_shape[2] * x_shape[3]:(ii+1) * x_shape[1] * x_shape[2] * x_shape[3]] = perm + ii * x_shape[1] * x_shape[2] * x_shape[3]
        # x_offset[:, :, :, :] = x[permth.long()].view(x_shape[0], x_shape[1], x_shape[2], x_shape[3])
        #
        # return self.dilated_conv(x_offset)
        # x_shape = x.size()  # [128, 3, 32, 32]
        # x = x.view(x_shape[0], x_shape[1] * x_shape[2] * x_shape[3])  # [128, 3*32*32]
        # x_offset = torch.empty(x_shape[0], x_shape[1], x_shape[2], x_shape[3]).cuda(cuda_number)
        # perm = torch.empty(0).float()
        # for i in range(x_shape[1]):
        #     a = torch.randperm(x_shape[2] * x_shape[3]) + i * x_shape[2] * x_shape[3]
        #     perm = torch.cat((perm, a.float()), 0)
        # x_offset[:, :, :, :] = x[:, perm.long()].view(x_shape[0], x_shape[1], x_shape[2], x_shape[3])
        # return self.dilated_conv(x_offset)
        x_shape = x.size()  # [128, 3, 32, 32]
        x = x.view(x_shape[0], -1)
        if self.indices is None:
            self._setup(x_shape[1], x_shape[2] * x_shape[3])
        for i in range(x_shape[1]):
            np.random.shuffle(self.indices[i])
        x = x[:, torch.from_numpy(self.indices)].view(x_shape)
        return self.dilated_conv(x)


class Dconv_shuffle_depthwise(nn.Module):
    """
    Deformable convolution with random shuffling of the feature map.
    Random shuffling only happened within each page independently.
    The sampling locations are generated for each forward pass during the training.
    """

    def __init__(self, inplane, outplane, kernel_size, stride, padding):
        super(Dconv_shuffle_depthwise, self).__init__()
        print('cifar Dconv_shuffle_depthwise is used')
        self.dilated_conv = nn.Conv2d(inplane, outplane, kernel_size=kernel_size, stride=stride, padding=padding,
                                      groups=inplane, bias=False)
        self.indices = None

    def _setup(self, inplane, spatial_size):
        self.indices = np.empty((inplane, spatial_size), dtype=np.int64)
        for i in range(inplane):
            self.indices[i, :] = np.arange(self.indices.shape[1]) + i * self.indices.shape[1]

    def forward(self, x):
        # x_shape = x.size()  # [128, 3, 32, 32]
        # x = x.view(x_shape[0] * x_shape[1] * x_shape[2] * x_shape[3], )  # [128, 3*32*32]
        # x_offset = torch.empty(x_shape[0], x_shape[1], x_shape[2], x_shape[3]).cuda(cuda_number)
        # # np.save('/nethome/yuefan/fanyue/dconv/x.npy', x.detach().cpu().numpy())
        # permth = torch.empty((x_shape[0] * x_shape[1] * x_shape[2] * x_shape[3], )).float()
        # for ii in range(x_shape[0]):
        #     perm = torch.empty((x_shape[1] * x_shape[2] * x_shape[3], )).float()
        #     for i in range(x_shape[1]):
        #         a = torch.randperm(x_shape[2] * x_shape[3]) + i * x_shape[2] * x_shape[3]
        #         perm[i * x_shape[2] * x_shape[3]:(i+1) * x_shape[2] * x_shape[3]] = a
        #     permth[ii * x_shape[1] * x_shape[2] * x_shape[3]:(ii+1) * x_shape[1] * x_shape[2] * x_shape[3]] = perm + ii * x_shape[1] * x_shape[2] * x_shape[3]
        # x_offset[:, :, :, :] = x[permth.long()].view(x_shape[0], x_shape[1], x_shape[2], x_shape[3])
        #
        # return self.dilated_conv(x_offset)
        # x_shape = x.size()  # [128, 3, 32, 32]
        # x = x.view(x_shape[0], x_shape[1] * x_shape[2] * x_shape[3])  # [128, 3*32*32]
        # x_offset = torch.empty(x_shape[0], x_shape[1], x_shape[2], x_shape[3]).cuda(cuda_number)
        # perm = torch.empty(0).float()
        # for i in range(x_shape[1]):
        #     a = torch.randperm(x_shape[2] * x_shape[3]) + i * x_shape[2] * x_shape[3]
        #     perm = torch.cat((perm, a.float()), 0)
        # x_offset[:, :, :, :] = x[:, perm.long()].view(x_shape[0], x_shape[1], x_shape[2], x_shape[3])
        # return self.dilated_conv(x_offset)
        x_shape = x.size()  # [128, 3, 32, 32]
        x = x.view(x_shape[0], -1)
        if self.indices is None:
            self._setup(x_shape[1], x_shape[2] * x_shape[3])
        for i in range(x_shape[1]):
            np.random.shuffle(self.indices[i])
        x = x[:, torch.from_numpy(self.indices)].view(x_shape)
        return self.dilated_conv(x)


class Dconv_localshuffle(nn.Module):
    """
    Deformable convolution with random shuffling of the feature map.
    We first split each feature map into patches, and shuffle only applies within a patch
    Random shuffling only happened within each page independently.
    The sampling locations are generated for each forward pass during the training.
    """

    def __init__(self, inplane, outplane, kernel_size, stride, padding, nrows, ncols):
        super(Dconv_localshuffle, self).__init__()
        print('cifar Dconv_local_shuffle is used')
        self.nrows = nrows
        self.ncols = ncols
        self.dilated_conv = nn.Conv2d(inplane, outplane, kernel_size=kernel_size, stride=stride, padding=padding,
                                      bias=False)

    def forward(self, x, ):
        x_shape = x.size()  # [128, 3, 32, 32]
        x = x.view(x_shape[0], x_shape[1] * x_shape[2] * x_shape[3])  # [128, 3*32*32]
        x_offset = torch.empty(x_shape[0], x_shape[1], x_shape[2], x_shape[3]).cuda(cuda_number)
        # np.save('/nethome/yuefan/fanyue/dconv/x.npy', x.detach().cpu().numpy())
        perm = torch.empty(0).float()
        for i in range(x_shape[1]):
            idx = torch.arange(x_shape[2] * x_shape[3]).view(x_shape[2], x_shape[3])
            idx = self.blockshaped(idx, self.nrows, self.ncols)
            for j in range(idx.size(0)):  # idx.size(0) is the number of blocks
                a = torch.randperm(self.nrows * self.ncols)
                idx[j] = idx[j][a]
            idx = idx.view(-1, self.nrows, self.ncols)
            idx = self.unblockshaped(idx, x_shape[2], x_shape[3]) + i * x_shape[2] * x_shape[3]
            perm = torch.cat((perm, idx.float()), 0)
        x_offset[:, :, :, :] = x[:, perm.long()].view(x_shape[0], x_shape[1], x_shape[2], x_shape[3])
        return self.dilated_conv(x_offset)

    def blockshaped(self, arr, nrows, ncols):
        """
        Return an array of shape (n, nrows, ncols) where
        n * nrows * ncols = arr.size

        If arr is a 2D array, the returned array should look like n subblocks with
        each subblock preserving the "physical" layout of arr.
        """
        h, w = arr.shape
        assert h % nrows == 0, "{} rows is not evenly divisble by {}".format(h, nrows)
        assert w % ncols == 0, "{} cols is not evenly divisble by {}".format(w, ncols)
        return (arr.view(h // nrows, nrows, -1, ncols)
                .permute(0, 2, 1, 3).contiguous()
                .view(-1, nrows * ncols))

    def unblockshaped(self, arr, h, w):
        """
        Return an array of shape (h, w) where
        h * w = arr.size

        If arr is of shape (n, nrows, ncols), n sublocks of shape (nrows, ncols),
        then the returned array preserves the "physical" layout of the sublocks.
        """
        n, nrows, ncols = arr.shape
        return (arr.view(h // nrows, -1, nrows, ncols)
                .permute(0, 2, 1, 3).contiguous()
                .view(-1, ))


class Dconv_cshuffle(nn.Module):
    """
    Deformable convolution with random shuffling of the feature map.
    This is the channel version of shuffle above.
    The sampling locations are generated for each forward pass during the training.
    """

    def __init__(self, inplane, outplane, kernel_size, stride, padding):
        super(Dconv_cshuffle, self).__init__()
        print('cifar Dconv_cshuffle is used')
        self.dilated_conv = nn.Conv2d(inplane, outplane, kernel_size=kernel_size, stride=stride, padding=padding,
                                      bias=False)

    def forward(self, x):
        x_shape = x.size()  # [128, 3, 32, 32]
        x = x.permute(0, 2, 3, 1)  # [128, 32, 32, 3]
        x = x.contiguous().view(x_shape[0], x_shape[1] * x_shape[2] * x_shape[3])  # [128, 32*32*3]

        x_offset = torch.empty(x_shape[0], x_shape[1], x_shape[2], x_shape[3]).cuda(cuda_number)
        # np.save('/nethome/yuefan/fanyue/dconv/x.npy', x.detach().cpu().numpy())
        perm = torch.empty(0).float()
        for i in range(x_shape[2] * x_shape[3]):
            a = torch.randperm(x_shape[1]) + i * x_shape[1]
            perm = torch.cat((perm, a.float()), 0)

        x_offset[:, :, :, :] = x[:, perm.long()].view(x_shape[0], x_shape[2], x_shape[3], x_shape[1]).permute(0, 3, 1,
                                                                                                              2)

        return self.dilated_conv(x_offset)


class Dconv_shuffleall(nn.Module):
    """
    Deformable convolution with random shuffling of the feature map.
    Random shuffle the whole feature map all together.
    The sampling locations are generated for each forward pass during the training.
    """

    def __init__(self, inplane, outplane, kernel_size, stride, padding):
        super(Dconv_shuffleall, self).__init__()
        print('cifar Dconv_shuffleall is used')
        self.dilated_conv = nn.Conv2d(inplane, outplane, kernel_size=kernel_size, stride=stride, padding=padding,
                                      bias=False)

    def forward(self, x):
        x_shape = x.size()  # [128, 3, 32, 32]
        x = x.view(x_shape[0], x_shape[1] * x_shape[2] * x_shape[3])  # [128, 3*32*32]
        x_offset = torch.empty(x_shape[0], x_shape[1] * x_shape[2] * x_shape[3]).cuda(cuda_number)
        # np.save('/nethome/yuefan/fanyue/dconv/weight.npy', self.dilated_conv.weight.detach().cpu().numpy())
        # np.save('/nethome/yuefan/fanyue/dconv/x.npy', x.detach().cpu().numpy())
        perm = torch.randperm(x_shape[1] * x_shape[2] * x_shape[3])
        x_offset[:, :] = x[:, perm]
        # x_offset[:, :] = 0

        x_offset = x_offset.view(x_shape[0], x_shape[1], x_shape[2], x_shape[3])

        return self.dilated_conv(x_offset)


class Dconv_horizontal(nn.Module):
    """
    Deformable convolution with random shuffling of the feature map.
    Random shuffle feature maps horizontally
    The sampling locations are generated for each forward pass during the training.
    """

    def __init__(self, inplane, outplane, kernel_size, stride, padding):
        super(Dconv_horizontal, self).__init__()
        print('Dconv_horizontal is used')
        self.dilated_conv = nn.Conv2d(inplane, outplane, kernel_size=kernel_size, stride=stride, padding=padding,
                                      bias=False)

    def forward(self, x):
        x_shape = x.size()  # [128, 3, 32, 32]
        x_offset = torch.empty(x_shape[0], x_shape[1], x_shape[2], x_shape[3]).cuda(cuda_number)
        perm = torch.randperm(x_shape[3])
        x_offset[:, :, :, :] = x[:, :, :, perm]

        return self.dilated_conv(x_offset)


class Dconv_vertical(nn.Module):
    """
    Deformable convolution with random shuffling of the feature map.
    Random shuffle feature maps vertically.
    The sampling locations are generated for each forward pass during the training.
    """

    def __init__(self, inplane, outplane, kernel_size, stride, padding):
        super(Dconv_vertical, self).__init__()
        print('Dconv_vertical is used')
        self.dilated_conv = nn.Conv2d(inplane, outplane, kernel_size=kernel_size, stride=stride, padding=padding,
                                      bias=False)

    def forward(self, x):
        x_shape = x.size()  # [128, 3, 32, 32]
        x_offset = torch.empty(x_shape[0], x_shape[1], x_shape[2], x_shape[3]).cuda(cuda_number)
        perm = torch.randperm(x_shape[2])
        x_offset[:, :, :, :] = x[:, :, perm, :]

        return self.dilated_conv(x_offset)


class Dconv_none(nn.Module):
    """
    This is just a Dconv wraper for a standard conv2d, it is only used for model resnet50_60_dconv
    """

    def __init__(self, inplane, outplane, kernel_size, stride, padding):
        super(Dconv_none, self).__init__()
        print('cifar Dconv_none is used')
        self.dilated_conv = nn.Conv2d(inplane, outplane, kernel_size=kernel_size, stride=stride, padding=padding,
                                      bias=False)

    def forward(self, x):
        # np.save('/nethome/yuefan/fanyue/dconv/fmbottel4.npy', x.detach().cpu().numpy())
        # np.save('/nethome/yuefan/fanyue/dconv/weight.npy', self.dilated_conv.weight.detach().cpu().numpy())
        return self.dilated_conv(x)


class Dconv_euc(nn.Module):
    """
    Deformable convolution with Euclidean similarity measure

    The key problems in our method are:
    1. How to obtain the new sampling locations.
        For now we choose 9 locations where the activations are most similar
        to the center one. And we discard the spatial distance between them.
        TODO: consider the spatial distance and use other similarity measure
    2. How to arrange a new squared layout for the upcoming convolution.
        For now we just go through all the activations from the left to the
        right and then top down.
        TODO: come up with a more descent arrangement
    """

    def __init__(self, height, width, inplane, outplane, win_range=5, kernel_size=3):
        """
        :param win_range: defines the size of the search window (for now it has to be odd)
        :param height: the height of the input feature map
        :param width: the width of the input feature map
        """
        super(Dconv_euc, self).__init__()
        self.h = height
        self.w = width
        self.win_range = win_range

        # offset dict is a dict of the same length of the number of activations in the input feature map.
        # Each key is a tuple denoting the location of the activation, the value is a list of the
        # indices of its neighbors, the length of the list is win_range * win_range.
        self.offset_dict = {}
        min_len = 10000  # the smallest length of the idx_list
        for key_i in range(self.h):
            for key_j in range(self.w):

                idx_list = []  # the list for the indices of the neighbors of the point (key_i, key_j)

                # go through all the positions nearby and add them to the list
                for i in range(self.h):
                    for j in range(self.w):
                        if np.abs(i - key_i) <= (self.win_range - 1) / 2 and np.abs(j - key_j) <= (
                                self.win_range - 1) / 2:
                            idx_list.append(self.loc_1D([i, j]))
                if len(idx_list) < min_len:
                    min_len = len(idx_list)
                self.offset_dict[(key_i, key_j)] = torch.tensor(idx_list).cuda(cuda_number)

        if min_len < kernel_size * kernel_size:
            raise Exception('the window size is too small to contain the kernel!')
        # since each location has its own offset, the re-arranged feature map should have a larger size than the x,
        # namely, x is 32x32 and kernel is 3x3, then x_offset should be 96x96. But only 32x32 locations out of 96x96
        # are needed for convolution. Thus a dilated convolution should be used. The stride should be the same as the
        # size of the kernel and the padding should be disabled. For now, only 3x3 convolution is supported.
        # TODO: support convolutions with any kernel size
        self.kernel_size = kernel_size
        self.dilated_conv = nn.Conv2d(inplane, outplane, kernel_size=self.kernel_size, stride=self.kernel_size,
                                      padding=0, bias=False)

        # the relative locations of an activation, those locations has to be filled, it is the same size of the kernel
        self.directions = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 0], [0, 1], [1, -1], [1, 0], [1, 1]]

    def forward(self, x):
        """
        Enumerate all (key, idx_list) pairs in the offset_dict.
        For each pair, first compute distance list, and then re-arrange x accordingly and assign x to the x_offset.
        Finally a dilated convolution performed on the x_offset.
        """
        x_shape = x.size()  # [128, 3, 32, 32]
        x = x.view(x_shape[0], x_shape[1], x_shape[2] * x_shape[3])  # [128, 3, 32*32]

        # assign the first n most similar activations from the neighbors to the squared neighbourhood of "key"
        x_offset = torch.empty(x_shape[0], x_shape[1], x_shape[2] * self.kernel_size,
                               x_shape[3] * self.kernel_size).cuda(cuda_number)

        for key, idx_list in self.offset_dict.items():
            tmpX1 = x[:, :, self.loc_1D(key)].view(x_shape[0], x_shape[1], 1)  # [128, 3, 1]
            tmpX2 = x[:, :, idx_list].view(x_shape[0], x_shape[1], len(idx_list))  # [128, 3, 7*7]

            # distance_list contains the cosine similarity between key and every location in idx_list.
            # It is batchified.
            dist_list = F.pairwise_distance(tmpX1, tmpX2, 2, 1e-6)  # [128, 7*7]

            # sort in an descending way because it is a similarity measure
            delll, orders = torch.sort(dist_list, dim=1, descending=True)  # TODO: this is wrong, should be ascent!!!
            # sort idx_list according to dist_list
            sorted_idx_list = torch.take(idx_list, orders)  # [128, 7*7]
            # take the first several smallest ones
            sorted_idx_list = sorted_idx_list[:, 0: self.kernel_size * self.kernel_size]  # [128, 3*3]
            # now sort the idx_list ascendingly
            sorted_idx_list, _ = torch.sort(sorted_idx_list, dim=1, descending=False)

            for i, relativ_loc in enumerate(self.directions):
                indd = sorted_idx_list[:, i]
                # print(indd)
                # print(x[torch.arange(0, 128, 1).long(), :, indd].size())
                x_offset[:, :, (key[0] * self.kernel_size + 1) + relativ_loc[0],
                (key[1] * self.kernel_size + 1) + relativ_loc[1]] = x[torch.arange(0, x_shape[0], 1).long(), :, indd]

        # apply dilated convolution so that skip the undefined locations
        x_offset = self.dilated_conv(x_offset)

        return x_offset

    def loc_2D(self, loc_1D):
        # turn a 1D coordinate into a 2D coordinate
        return [loc_1D // self.w, loc_1D % self.w]

    def loc_1D(self, loc_2D):
        # turn a 2D coordinate into a 1D coordinate
        return loc_2D[0] * self.w + loc_2D[1]


class DConv1Dai_share(nn.Conv2d):
    """DConv1Dai_share

    The modified version of the original deformable convolution.
    The sampling location is shared along channels. Usage is the same as DConv1Dai.

    It improves over the original dconv.

    TODO: support convolutions with any kernel size
    """

    def __init__(self, filters, init_normal_stddev=0.01, **kwargs):
        """Init

        Parameters
        ----------
        filters : int
            Number of channel of the input feature map
        init_normal_stddev : float
            Normal kernel initialization
        **kwargs:
            Pass to superclass. See Con2d layer in pytorch
        """
        self.filters = filters
        self._grid_param = None
        super(DConv1Dai_share, self).__init__(self.filters, self.filters, 3, padding=1, bias=False, **kwargs)
        self.weight.data.copy_(self._init_weights(self.weight, init_normal_stddev))

    def forward(self, x):
        """Return the deformed featured map"""
        x_shape = x.size()  # [128, 3, 32, 32]

        offset = super(DConv1Dai_share, self).forward(x)  # [128, 3, 32, 32]
        # a = offsets.contiguous()[0,:,:,:]
        # a = a.view(3, 32*32, 2)
        # print()
        # print(a)
        # (b, c*2, h, w)
        offset = offset.view(x_shape[0] * x_shape[1], x_shape[2] * x_shape[3])  # [128*3, 32*32]
        offset = offset.unsqueeze(-1)  # [128*3, 32*32, 1]
        offsets = torch.cat((offset.contiguous(), offset.contiguous()), 2)  # [128*3, 32, 32, 2]
        offsets = offsets.view(x_shape[0], 2 * x_shape[1], x_shape[2], x_shape[3])  # [128, 2*3, 32, 32]
        # (b*c, h, w, 2)
        offsets = self._to_bc_h_w_2(offsets, x_shape)

        # (b*c, h, w)
        x = self._to_bc_h_w(x, x_shape)

        # X_offset: (b*c, h*w)    grid:(b*c, h*w, 2)
        # grid contains b*c same 2D arrays, which is
        #  [[0 0]
        #   [0 1]
        #   [0 2]
        #   [1 0]
        #   [1 1]
        #   [1 2]]
        # for h=2, w=3
        x_offset = th_batch_map_offsets(x, offsets, grid=self._get_grid(self, x))
        # x_offset: (b, c, h, w)
        x_offset = self._to_b_c_h_w(x_offset, x_shape)
        return x_offset

    @staticmethod
    def _get_grid(self, x):
        batch_size, input_size = x.size(0), (x.size(1), x.size(2))
        dtype, cuda = x.data.type(), x.data.is_cuda
        if self._grid_param == (batch_size, input_size, dtype, cuda):
            return self._grid
        self._grid_param = (batch_size, input_size, dtype, cuda)
        self._grid = th_generate_grid(batch_size, input_size, dtype, cuda)
        return self._grid

    @staticmethod
    def _init_weights(weights, std):
        fan_out = weights.size(0)
        fan_in = weights.size(1) * weights.size(2) * weights.size(3)
        w = np.random.normal(0.0, std, (fan_out, fan_in))
        return torch.from_numpy(w.reshape(weights.size()))

    @staticmethod
    def _to_bc_h_w_2(x, x_shape):
        """(b, 2c, h, w) -> (b*c, h, w, 2)"""
        x = x.contiguous().view(-1, int(x_shape[2]), int(x_shape[3]), 2)
        return x

    @staticmethod
    def _to_bc_h_w(x, x_shape):
        """(b, c, h, w) -> (b*c, h, w)"""
        x = x.contiguous().view(-1, int(x_shape[2]), int(x_shape[3]))
        return x

    @staticmethod
    def _to_b_c_h_w(x, x_shape):
        """(b*c, h, w) -> (b, c, h, w)"""
        x = x.contiguous().view(-1, int(x_shape[1]), int(x_shape[2]), int(x_shape[3]))
        return x


class DConv1Dai(nn.Conv2d):
    """DConv1Dai

    Convolutional layer responsible for learning the 2D offsets and output the
    deformed feature map using bilinear interpolation

    Note that this layer does not perform convolution on the deformed feature
    map. Usage:
    self.offset = DConv1Dai(plane1)
    self.conv2 = nn.Conv2d(plane1, plane2, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) / 2, bias=False)

    out = self.offset(out)
    out = self.conv2(out)
    """

    def __init__(self, filters, init_normal_stddev=0.01, **kwargs):
        """Init

        Parameters
        ----------
        filters : int
            Number of channel of the input feature map
        init_normal_stddev : float
            Normal kernel initialization
        **kwargs:
            Pass to superclass. See Con2d layer in pytorch
        """
        self.filters = filters
        self._grid_param = None
        super(DConv1Dai, self).__init__(self.filters, self.filters * 2, 3, padding=1, bias=False, **kwargs)
        self.weight.data.copy_(self._init_weights(self.weight, init_normal_stddev))

    def forward(self, x):
        """Return the deformed featured map"""
        x_shape = x.size()  # [128, 3, 32, 32]

        # (b, c*2, h, w)
        offsets = super(DConv1Dai, self).forward(x)  # [128, 6, 32, 32]

        # (b*c, h, w, 2)
        offsets = self._to_bc_h_w_2(offsets, x_shape)

        # (b*c, h, w)
        x = self._to_bc_h_w(x, x_shape)

        # X_offset: (b*c, h*w)    grid:(b*c, h*w, 2)
        # grid contains b*c same 2D arrays, which is
        #  [[0 0]
        #   [0 1]
        #   [0 2]
        #   [1 0]
        #   [1 1]
        #   [1 2]]
        # for h=2, w=3
        x_offset = th_batch_map_offsets(x, offsets, grid=self._get_grid(self, x))
        # x_offset: (b, c, h, w)
        x_offset = self._to_b_c_h_w(x_offset, x_shape)
        return x_offset

    @staticmethod
    def _get_grid(self, x):
        batch_size, input_size = x.size(0), (x.size(1), x.size(2))
        dtype, cuda = x.data.type(), x.data.is_cuda
        if self._grid_param == (batch_size, input_size, dtype, cuda):
            return self._grid
        self._grid_param = (batch_size, input_size, dtype, cuda)
        self._grid = th_generate_grid(batch_size, input_size, dtype, cuda)
        return self._grid

    @staticmethod
    def _init_weights(weights, std):
        fan_out = weights.size(0)
        fan_in = weights.size(1) * weights.size(2) * weights.size(3)
        w = np.random.normal(0.0, std, (fan_out, fan_in))
        return torch.from_numpy(w.reshape(weights.size()))

    @staticmethod
    def _to_bc_h_w_2(x, x_shape):
        """(b, 2c, h, w) -> (b*c, h, w, 2)"""
        x = x.contiguous().view(-1, int(x_shape[2]), int(x_shape[3]), 2)
        return x

    @staticmethod
    def _to_bc_h_w(x, x_shape):
        """(b, c, h, w) -> (b*c, h, w)"""
        x = x.contiguous().view(-1, int(x_shape[2]), int(x_shape[3]))
        return x

    @staticmethod
    def _to_b_c_h_w(x, x_shape):
        """(b*c, h, w) -> (b, c, h, w)"""
        x = x.contiguous().view(-1, int(x_shape[1]), int(x_shape[2]), int(x_shape[3]))
        return x


def th_flatten(a):
    """Flatten tensor"""
    return a.contiguous().view(a.nelement())


def th_repeat(a, repeats, axis=0):
    """Torch version of np.repeat for 1D"""
    assert len(a.size()) == 1
    return th_flatten(torch.transpose(a.repeat(repeats, 1), 0, 1))


def np_repeat_2d(a, repeats):
    """Tensorflow version of np.repeat for 2D"""

    assert len(a.shape) == 2
    a = np.expand_dims(a, 0)
    a = np.tile(a, [repeats, 1, 1])
    return a


def th_gather_2d(input, coords):
    inds = coords[:, 0] * input.size(1) + coords[:, 1]
    x = torch.index_select(th_flatten(input), 0, inds)
    return x.view(coords.size(0))


def th_map_coordinates(input, coords, order=1):
    """Tensorflow verion of scipy.ndimage.map_coordinates
    Note that coords is transposed and only 2D is supported
    Parameters
    ----------
    input : tf.Tensor. shape = (s, s)
    coords : tf.Tensor. shape = (n_points, 2)
    """

    assert order == 1
    input_size = input.size(0)

    coords = torch.clamp(coords, 0, input_size - 1)
    coords_lt = coords.floor().long()
    coords_rb = coords.ceil().long()
    coords_lb = torch.stack([coords_lt[:, 0], coords_rb[:, 1]], 1)
    coords_rt = torch.stack([coords_rb[:, 0], coords_lt[:, 1]], 1)

    vals_lt = th_gather_2d(input, coords_lt.detach())
    vals_rb = th_gather_2d(input, coords_rb.detach())
    vals_lb = th_gather_2d(input, coords_lb.detach())
    vals_rt = th_gather_2d(input, coords_rt.detach())

    coords_offset_lt = coords - coords_lt.type(coords.data.type())

    vals_t = vals_lt + (vals_rt - vals_lt) * coords_offset_lt[:, 0]
    vals_b = vals_lb + (vals_rb - vals_lb) * coords_offset_lt[:, 0]
    mapped_vals = vals_t + (vals_b - vals_t) * coords_offset_lt[:, 1]
    return mapped_vals


def sp_batch_map_coordinates(inputs, coords):
    """Reference implementation for batch_map_coordinates"""
    coords = coords.clip(0, inputs.shape[1] - 1)
    mapped_vals = np.array([
        sp_map_coordinates(input, coord.T, mode='nearest', order=1)
        for input, coord in zip(inputs, coords)
    ])
    return mapped_vals


def th_batch_map_coordinates(input, coords, order=1):
    """Batch version of th_map_coordinates
    Only supports 2D feature maps
    Parameters
    ----------
    input : tf.Tensor. shape = (bc, s, s)
    coords : tf.Tensor. shape = (bc, s*s, 2)
    Returns
    -------
    tf.Tensor. shape = (bc, s, s)
    """

    batch_size = input.size(0)
    input_size = input.size(1)
    n_coords = coords.size(1)  # n_coords is the h*w
    # print('sadsadsaddsadsadsadsadsadsadsa')
    # print(coords)
    # print('sadsadsaddsadsadsadsadsadsadsa')
    coords = torch.clamp(coords, 0, input_size - 1)  # the range of the offset can cover the whole image
    # turn the fractional locations into the 4 nearest integer locations
    coords_lt = coords.floor().long()
    coords_rb = coords.ceil().long()
    coords_lb = torch.stack([coords_lt[..., 0], coords_rb[..., 1]], 2)
    coords_rt = torch.stack([coords_rb[..., 0], coords_lt[..., 1]], 2)
    idx = th_repeat(torch.arange(0, batch_size), n_coords).long()
    # print('sadsadsaddsadsadsadsadsadsadsa')
    # print(idx.size())
    # print('sadsadsaddsadsadsadsadsadsadsa')
    idx = Variable(idx, requires_grad=False)
    if input.is_cuda:
        idx = idx.cuda(cuda_number)

    def _get_vals_by_coords(input, coords):
        indices = torch.stack([
            idx, th_flatten(coords[..., 0]), th_flatten(coords[..., 1])
        ], 1)
        inds = indices[:, 0] * input.size(1) * input.size(2) + indices[:, 1] * input.size(2) + indices[:, 2]
        vals = th_flatten(input).index_select(0, inds)
        vals = vals.view(batch_size, n_coords)
        return vals

    vals_lt = _get_vals_by_coords(input, coords_lt.detach())
    vals_rb = _get_vals_by_coords(input, coords_rb.detach())
    vals_lb = _get_vals_by_coords(input, coords_lb.detach())
    vals_rt = _get_vals_by_coords(input, coords_rt.detach())

    coords_offset_lt = coords - coords_lt.type(coords.data.type())
    vals_t = coords_offset_lt[..., 0] * (vals_rt - vals_lt) + vals_lt
    vals_b = coords_offset_lt[..., 0] * (vals_rb - vals_lb) + vals_lb
    mapped_vals = coords_offset_lt[..., 1] * (vals_b - vals_t) + vals_t
    return mapped_vals


def sp_batch_map_offsets(input, offsets):
    """Reference implementation for tf_batch_map_offsets"""

    batch_size = input.shape[0]
    input_size = input.shape[1]

    offsets = offsets.reshape(batch_size, -1, 2)
    grid = np.stack(np.mgrid[:input_size, :input_size], -1).reshape(-1, 2)
    grid = np.repeat([grid], batch_size, axis=0)
    coords = offsets + grid
    coords = coords.clip(0, input_size - 1)

    mapped_vals = sp_batch_map_coordinates(input, coords)
    return mapped_vals


def th_generate_grid(batch_size, input_size, dtype, cuda):
    grid = np.meshgrid(
        range(input_size[0]), range(input_size[1]), indexing='ij'
    )
    grid = np.stack(grid, axis=-1)
    grid = grid.reshape(-1, 2)

    grid = np_repeat_2d(grid, batch_size)
    grid = torch.from_numpy(grid).type(dtype)
    if cuda:
        grid = grid.cuda(cuda_number)
    return Variable(grid, requires_grad=False)


def th_batch_map_offsets(input, offsets, grid=None, order=1):
    """Batch map offsets into input
    Parameters
    ---------
    input : torch.Tensor. shape = (bc, s, s)
    offsets: torch.Tensor. shape = (bc, s, s, 2)
    grid: (b*c, h*w, 2), which is the x-y location
    Returns
    -------
    torch.Tensor. shape = (bc, s, s)
    """
    batch_size = input.size(0)
    input_size = [input.size(1), input.size(2)]

    # (bc, h*w, 2)
    offsets = offsets.view(batch_size, -1, 2)
    if grid is None:
        # grid:(b*c, h*w, 2)
        grid = th_generate_grid(batch_size, input_size, offsets.data.type(), offsets.data.is_cuda)
    # (b*c, h*w, 2)
    coords = offsets + grid
    # (b*c, h*w)| (b*c, h*w), (b*c, h*w, c)
    mapped_vals = th_batch_map_coordinates(input, coords)

    return mapped_vals
