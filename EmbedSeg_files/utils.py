import os
import threading
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from PIL import Image
from matplotlib.patches import Ellipse
from skimage.feature import peak_local_max


def to_cuda(im_numpy):
    """
    Converts 2D Numpy Image on CPU to PyTorch Tensor on GPU, with an extra dimension

    Parameters
    -------

    im_numpy: numpy array (YX)


    Returns
    -------
        Pytorch Tensor (1YX)

    """
    im_numpy = im_numpy[np.newaxis, ...]
    return torch.from_numpy(im_numpy).float().cuda()


def to_numpy(im_cuda):
    """
    Converts PyTorch Tensor on GPU to Numpy Array on CPU

    Parameters
    -------

    im_cuda: PyTorch tensor


    Returns
    -------
        numpy array

    """
    return im_cuda.cpu().detach().numpy()


def process_flips(im_numpy):
    """
    Converts the model output (5YX) so that y-offset is correctly handled
    (x-offset, y-offset, x-margin bandwidth, y-margin bandwidth, seediness score)

    Parameters
    -------

    im_numpy: Numpy Array (5YX)


    Returns
    -------
    im_numpy_correct: Numpy Array (5YX)

    """
    im_numpy_correct = im_numpy
    im_numpy_correct[0, 1, ...] = -1 * im_numpy[
        0, 1, ...]  # because flipping is always along y-axis, so only the y-offset gets affected
    return im_numpy_correct


def apply_tta_2d(im, model):
    """
    Apply Test Time Augmentation for 2D Images

    Parameters
    -------

    im: Numpy Array (1CYX)
    model: PyTorch Model

    Returns
    -------
    PyTorch Tensor on GPU (15YX)

    """
    im_numpy = im.cpu().detach().numpy()  # BCYX
    im0 = im_numpy[0, ...]  # remove batch dimension, now CYX
    im1 = np.rot90(im0, 1, (1, 2))
    im2 = np.rot90(im0, 2, (1, 2))
    im3 = np.rot90(im0, 3, (1, 2))
    im4 = np.flip(im0, 1)
    im5 = np.flip(im1, 1)
    im6 = np.flip(im2, 1)
    im7 = np.flip(im3, 1)

    im0_cuda = to_cuda(im0)  # BCYX
    im1_cuda = to_cuda(np.ascontiguousarray(im1))
    im2_cuda = to_cuda(np.ascontiguousarray(im2))
    im3_cuda = to_cuda(np.ascontiguousarray(im3))
    im4_cuda = to_cuda(np.ascontiguousarray(im4))
    im5_cuda = to_cuda(np.ascontiguousarray(im5))
    im6_cuda = to_cuda(np.ascontiguousarray(im6))
    im7_cuda = to_cuda(np.ascontiguousarray(im7))

    output0 = model(im0_cuda)
    output1 = model(im1_cuda)
    output2 = model(im2_cuda)
    output3 = model(im3_cuda)
    output4 = model(im4_cuda)
    output5 = model(im5_cuda)
    output6 = model(im6_cuda)
    output7 = model(im7_cuda)

    # de-transform outputs
    output0_numpy = to_numpy(output0)
    output1_numpy = to_numpy(output1)
    output2_numpy = to_numpy(output2)
    output3_numpy = to_numpy(output3)
    output4_numpy = to_numpy(output4)
    output5_numpy = to_numpy(output5)
    output6_numpy = to_numpy(output6)
    output7_numpy = to_numpy(output7)

    # invert rotations and flipping

    output1_numpy = np.rot90(output1_numpy, 1, (3, 2))
    output2_numpy = np.rot90(output2_numpy, 2, (3, 2))
    output3_numpy = np.rot90(output3_numpy, 3, (3, 2))
    output4_numpy = np.flip(output4_numpy, 2)
    output5_numpy = np.flip(output5_numpy, 2)
    output5_numpy = np.rot90(output5_numpy, 1, (3, 2))
    output6_numpy = np.flip(output6_numpy, 2)
    output6_numpy = np.rot90(output6_numpy, 2, (3, 2))
    output7_numpy = np.flip(output7_numpy, 2)
    output7_numpy = np.rot90(output7_numpy, 3, (3, 2))

    # have to also process the offsets and covariance sensibly
    output0_numpy_correct = output0_numpy

    # note rotations are always [cos(theta) sin(theta); -sin(theta) cos(theta)]
    output1_numpy_correct = np.zeros_like(output1_numpy)
    output1_numpy_correct[0, 0, ...] = -output1_numpy[0, 1, ...]
    output1_numpy_correct[0, 1, ...] = output1_numpy[0, 0, ...]
    output1_numpy_correct[0, 2, ...] = output1_numpy[0, 3, ...]
    output1_numpy_correct[0, 3, ...] = output1_numpy[0, 2, ...]
    output1_numpy_correct[0, 4, ...] = output1_numpy[0, 4, ...]

    output2_numpy_correct = np.zeros_like(output2_numpy)
    output2_numpy_correct[0, 0, ...] = -output2_numpy[0, 0, ...]
    output2_numpy_correct[0, 1, ...] = -output2_numpy[0, 1, ...]
    output2_numpy_correct[0, 2, ...] = output2_numpy[0, 2, ...]
    output2_numpy_correct[0, 3, ...] = output2_numpy[0, 3, ...]
    output2_numpy_correct[0, 4, ...] = output2_numpy[0, 4, ...]

    output3_numpy_correct = np.zeros_like(output3_numpy)
    output3_numpy_correct[0, 0, ...] = output3_numpy[0, 1, ...]
    output3_numpy_correct[0, 1, ...] = -output3_numpy[0, 0, ...]
    output3_numpy_correct[0, 2, ...] = output3_numpy[0, 3, ...]
    output3_numpy_correct[0, 3, ...] = output3_numpy[0, 2, ...]
    output3_numpy_correct[0, 4, ...] = output3_numpy[0, 4, ...]

    output4_numpy_correct = process_flips(output4_numpy)

    output5_numpy_flipped = process_flips(output5_numpy)
    output5_numpy_correct = np.zeros_like(output5_numpy_flipped)
    output5_numpy_correct[0, 0, ...] = -output5_numpy_flipped[0, 1, ...]
    output5_numpy_correct[0, 1, ...] = output5_numpy_flipped[0, 0, ...]
    output5_numpy_correct[0, 2, ...] = output5_numpy_flipped[0, 3, ...]
    output5_numpy_correct[0, 3, ...] = output5_numpy_flipped[0, 2, ...]
    output5_numpy_correct[0, 4, ...] = output5_numpy_flipped[0, 4, ...]

    output6_numpy_flipped = process_flips(output6_numpy)
    output6_numpy_correct = np.zeros_like(output6_numpy_flipped)
    output6_numpy_correct[0, 0, ...] = -output6_numpy_flipped[0, 0, ...]
    output6_numpy_correct[0, 1, ...] = -output6_numpy_flipped[0, 1, ...]
    output6_numpy_correct[0, 2, ...] = output6_numpy_flipped[0, 2, ...]
    output6_numpy_correct[0, 3, ...] = output6_numpy_flipped[0, 3, ...]
    output6_numpy_correct[0, 4, ...] = output6_numpy_flipped[0, 4, ...]

    output7_numpy_flipped = process_flips(output7_numpy)
    output7_numpy_correct = np.zeros_like(output7_numpy_flipped)
    output7_numpy_correct[0, 0, ...] = output7_numpy_flipped[0, 1, ...]
    output7_numpy_correct[0, 1, ...] = -output7_numpy_flipped[0, 0, ...]
    output7_numpy_correct[0, 2, ...] = output7_numpy_flipped[0, 3, ...]
    output7_numpy_correct[0, 3, ...] = output7_numpy_flipped[0, 2, ...]
    output7_numpy_correct[0, 4, ...] = output7_numpy_flipped[0, 4, ...]

    output = np.concatenate((output0_numpy_correct, output1_numpy_correct, output2_numpy_correct, output3_numpy_correct,
                             output4_numpy_correct, output5_numpy_correct, output6_numpy_correct,
                             output7_numpy_correct), 0)
    output = np.mean(output, 0, keepdims=True)  # 1 5 Y X
    return torch.from_numpy(output).float().cuda()


def invert_one_hot(image):
    """Inverts a one-hot label mask.

                Parameters
                ----------
                image : numpy array (I x H x W)
                    Label mask present in one-hot fashion (i.e. with 0s and 1s and multiple z slices)
                    here `I` is the number of GT or predicted objects

                Returns
                -------
                numpy array (H x W)
                    A flattened label mask with objects labelled from 1 ... I
                """
    instance = np.zeros((image.shape[1], image.shape[2]), dtype="uint16")
    for z in range(image.shape[0]):
        instance = np.where(image[z] > 0, instance + z + 1, instance)
        # TODO - Alternate ways of inverting one-hot label masks would exist !!
    return instance


class Cluster:
    """
            A class used to cluster pixel embeddings in 2D

            Attributes
            ----------
            xym :  float (2, W, D)
                    pixel coordinates of tile /grid

            one_hot : bool
                    Should be set to True, if the GT label masks are present in a one-hot encoded fashion

            grid_x: int
                    Length (width) of grid / tile

            grid_y: int
                    Height of grid / tile

            pixel_x: float
                    if grid_x = 1000 and pixel_x = 1.0, then the pixel spacing along the x direction is pixel_x/(grid_x-1) = 1/999
            pixel_y: float
                    if grid_y = 1000 and pixel_y = 1.0, then the pixel spacing along the y direction is pixel_y/(grid_y-1) = 1/999


            Methods
            -------
            __init__: Initializes an object of class `Cluster_3d`

            cluster_with_gt: use the predicted spatial embeddings from all pixels belonging to the GT label mask
                        to identify the predicted cluster (used during training and validation)

            cluster:    use the  predicted spatial embeddings from all pixels in the test image.
                        Employs `fg_thresh` and `seed_thresh`
            cluster_local_maxima: use the  predicted spatial embeddings from all pixels in the test image.
                        Employs only `fg_thresh`
            """

    def __init__(self, grid_y, grid_x, pixel_y, pixel_x, one_hot=False):

        """
           Parameters
           ----------
           xym :  float (2, W, D)
                    pixel coordinates of tile /grid

            one_hot : bool
                    Should be set to True, if the GT label masks are present in a one-hot encoded fashion

            grid_x: int
                    Length (width) of grid / tile

            grid_y: int
                    Height of grid / tile

            pixel_x: float
                    if grid_x = 1000 and pixel_x = 1.0, then the pixel spacing along the x direction is pixel_x/(grid_x-1) = 1/999
            pixel_y: float
                    if grid_y = 1000 and pixel_y = 1.0, then the pixel spacing along the y direction is pixel_y/(grid_y-1) = 1/999

                   """
        xm = torch.linspace(0, pixel_x, grid_x).view(1, 1, -1).expand(1, grid_y, grid_x)
        ym = torch.linspace(0, pixel_y, grid_y).view(1, -1, 1).expand(1, grid_y, grid_x)
        xym = torch.cat((xm, ym), 0)

        self.xym = xym.cuda()
        self.one_hot = one_hot
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.pixel_x = pixel_x
        self.pixel_y = pixel_y

    def cluster_with_gt(self, prediction, gt_instance, n_sigma=2, ):
        """
            Parameters
            ----------
            prediction :  PyTorch Tensor
                    Model Prediction (5, H, W)
            gt_instance : PyTorch Tensor
                    Ground Truth Instance Segmentation Label Map

            n_sigma: int, default = 2
                    Number of dimensions in Raw Image
            Returns
            ----------
            instance: PyTorch Tensor (H, W)
                    instance segmentation
       """

        height, width = prediction.size(1), prediction.size(2)

        xym_s = self.xym[:, 0:height, 0:width]  # 2 x h x w

        spatial_emb = torch.tanh(prediction[0:2]) + xym_s  # 2 x h x w
        sigma = prediction[2:2 + n_sigma]  # n_sigma x h x w

        instance_map = torch.zeros(height, width).short().cuda()
        if (self.one_hot):
            unique_instances = torch.arange(gt_instance.size(0))
        else:
            unique_instances = gt_instance.unique()
            unique_instances = unique_instances[unique_instances != 0]

        for id in unique_instances:
            if (self.one_hot):
                mask = gt_instance[id].eq(1).view(1, height, width)
            else:
                mask = gt_instance.eq(id).view(1, height, width)

            center = spatial_emb[mask.expand_as(spatial_emb)].view(
                2, -1).mean(1).view(2, 1, 1)  # 2 x 1 x 1

            s = sigma[mask.expand_as(sigma)].view(n_sigma, -1).mean(1).view(n_sigma, 1, 1)

            s = torch.exp(s * 10)  # n_sigma x 1 x 1 #
            dist = torch.exp(-1 * torch.sum(torch.pow(spatial_emb - center, 2) * s, 0))
            proposal = (dist > 0.5)
            if (self.one_hot):
                instance_map[proposal] = id.item() + 1
            else:
                instance_map[proposal] = id.item()

        return instance_map

    def cluster_local_maxima(self, prediction, n_sigma=2, fg_thresh=0.5, min_mask_sum=0, min_unclustered_sum=0,
                             min_object_size=36):

        """
            Parameters
            ----------
            prediction :  PyTorch Tensor
                    Model Prediction (5, H, W)
            n_sigma: int, default = 2
                    Number of dimensions in Raw Image
            fg_thresh: float, default=0.5
                    Foreground Threshold defines which pixels are considered to the form the Foreground
                    and which would need to be clustered into unique objects
            min_mask_sum: int
                    Only start creating instances, if there are at least `min_mask_sum` pixels in foreground!
            min_unclustered_sum: int
                    Stop when the number of seed candidates are less than `min_unclustered_sum`
            min_object_size: int
                Predicted Objects below this threshold are ignored

            Returns
            ----------
            instance: PyTorch Tensor (H, W)
                    instance segmentation
       """

        from scipy.ndimage import gaussian_filter
        height, width = prediction.size(1), prediction.size(2)
        xym_s = self.xym[:, 0:height, 0:width]
        spatial_emb = torch.tanh(prediction[0:2]) + xym_s  # 2 x h x w
        sigma = prediction[2:2 + n_sigma]  # n_sigma x h x w
        seed_map = torch.sigmoid(prediction[2 + n_sigma:2 + n_sigma + 1])  # 1 x h x w
        instance_map = torch.zeros(height, width).short()
        # instances = []  # list
        count = 1
        mask_fg = seed_map > fg_thresh
        seed_map_cpu = seed_map.cpu().detach().numpy()
        seed_map_cpu_smooth = gaussian_filter(seed_map_cpu[0], sigma=1)
        coords = peak_local_max(seed_map_cpu_smooth)
        zeros = np.zeros((coords.shape[0], 1), dtype=np.uint8)
        coords = np.hstack((zeros, coords))

        mask_local_max_cpu = np.zeros(seed_map_cpu.shape, dtype=np.bool)
        mask_local_max_cpu[tuple(coords.T)] = True
        mask_local_max = torch.from_numpy(mask_local_max_cpu).bool().cuda()

        mask_seed = mask_fg * mask_local_max
        if mask_fg.sum() > min_mask_sum:
            spatial_emb_fg_masked = spatial_emb[mask_fg.expand_as(spatial_emb)].view(n_sigma, -1)  # fg candidate pixels
            spatial_emb_seed_masked = spatial_emb[mask_seed.expand_as(spatial_emb)].view(n_sigma,
                                                                                         -1)  # seed candidate pixels

            sigma_seed_masked = sigma[mask_seed.expand_as(sigma)].view(n_sigma, -1)  # sigma for seed candidate pixels
            seed_map_seed_masked = seed_map[mask_seed].view(1, -1)  # seediness for seed candidate pixels

            unprocessed = torch.ones(mask_seed.sum()).short().cuda()  # unclustered seed candidate pixels
            unclustered = torch.ones(mask_fg.sum()).short().cuda()  # unclustered fg candidate pixels
            instance_map_masked = torch.zeros(mask_fg.sum()).short().cuda()
            while (unprocessed.sum() > min_unclustered_sum):
                seed = (seed_map_seed_masked * unprocessed.float()).argmax().item()
                center = spatial_emb_seed_masked[:, seed:seed + 1]
                unprocessed[seed] = 0
                s = torch.exp(sigma_seed_masked[:, seed:seed + 1] * 10)
                dist = torch.exp(-1 * torch.sum(torch.pow(spatial_emb_fg_masked - center, 2) * s, 0))
                proposal = (dist > 0.5).squeeze()
                if proposal.sum() > min_object_size:
                    if unclustered[proposal].sum().float() / proposal.sum().float() > 0.5:
                        instance_map_masked[proposal.squeeze()] = count
                        count += 1
                        unclustered[proposal] = 0
                        # note the line above increases false positives, tab back twice to show less objects!
                        # The reason I leave it like so is because the penalty on false-negative nodes is `10` while
                        # penalty on false-positive nodes is `1`.
            instance_map[mask_fg.squeeze().cpu()] = instance_map_masked.cpu()
        return instance_map

    def cluster(self, prediction, n_sigma=2, seed_thresh=0.9, fg_thresh=0.5, min_mask_sum=0, min_unclustered_sum=0,
                min_object_size=36):

        """
            Parameters
            ----------
            prediction :  PyTorch Tensor
                    Model Prediction (5, H, W)
            n_sigma: int, default = 2
                    Number of dimensions in Raw Image
            seed_thresh : float, default=0.9
                    Seediness Threshold defines which pixels are considered to identify object centres
            fg_thresh: float, default=0.5
                    Foreground Threshold defines which pixels are considered to the form the Foreground
                    and which would need to be clustered into unique objects
            min_mask_sum: int
                    Only start creating instances, if there are at least `min_mask_sum` pixels in foreground!
            min_unclustered_sum: int
                    Stop when the number of seed candidates are less than `min_unclustered_sum`
            min_object_size: int
                Predicted Objects below this threshold are ignored

            Returns
            ----------
            instance: PyTorch Tensor (H, W)
                    instance segmentation
           """

        height, width = prediction.size(1), prediction.size(2)
        xym_s = self.xym[:, 0:height, 0:width]

        spatial_emb = torch.tanh(prediction[0:2]) + xym_s  # 2 x h x w

        sigma = prediction[2:2 + n_sigma]  # n_sigma x h x w
        seed_map = torch.sigmoid(prediction[2 + n_sigma:2 + n_sigma + 1])  # 1 x h x w

        instance_map = torch.zeros(height, width).short()
        instances = []  # list

        count = 1
        mask = seed_map > fg_thresh

        if mask.sum() > min_mask_sum:

            spatial_emb_masked = spatial_emb[mask.expand_as(spatial_emb)].view(n_sigma, -1)
            sigma_masked = sigma[mask.expand_as(sigma)].view(n_sigma, -1)
            seed_map_masked = seed_map[mask].view(1, -1)

            unclustered = torch.ones(mask.sum()).short().cuda()
            instance_map_masked = torch.zeros(mask.sum()).short().cuda()

            while (unclustered.sum() > min_unclustered_sum):
                seed = (seed_map_masked * unclustered.float()).argmax().item()
                seed_score = (seed_map_masked * unclustered.float()).max().item()
                if seed_score < seed_thresh:
                    break
                center = spatial_emb_masked[:, seed:seed + 1]
                unclustered[seed] = 0

                s = torch.exp(sigma_masked[:, seed:seed + 1] * 10)
                dist = torch.exp(-1 * torch.sum(torch.pow(spatial_emb_masked - center, 2) * s, 0))

                proposal = (dist > 0.5).squeeze()

                if proposal.sum() > min_object_size:
                    if unclustered[proposal].sum().float() / proposal.sum().float() > 0.5:
                        instance_map_masked[proposal.squeeze()] = count
                        instance_mask = torch.zeros(height, width).short()
                        instance_mask[mask.squeeze().cpu()] = proposal.short().cpu()
                        # center_image = torch.zeros(height, width).short()
                        #
                        # center[0] = int(degrid(center[0].cpu().detach().numpy(), self.grid_x, self.pixel_x))
                        # center[1] = int(degrid(center[1].cpu().detach().numpy(), self.grid_y, self.pixel_y))
                        # center_image[np.clip(int(center[1].item()), 0, height - 1), np.clip(int(center[0].item()), 0,
                        #                                                               width - 1)] = True
                        # instances.append(
                        #     {'mask': instance_mask.squeeze() * 255, 'score': seed_score,
                        #      'center-image': center_image})
                        count += 1
                        instances.append((instance_mask.squeeze() * 1).detach().cpu().numpy())

                        # if count>2:
                        #     import matplotlib.pyplot as plt
                        #     # plt.imshow(proposal[0, :, :].detach().cpu().numpy())
                        #     plt.imshow((instance_mask.squeeze() * 255).detach().cpu().numpy())
                        #     plt.show()

                unclustered[proposal] = 0

            instance_map[mask.squeeze().cpu()] = instance_map_masked.cpu()
            # print('instance_map_masked', instance_map_masked.size())
            # print('instance_map', instance_map.size())

        # import matplotlib.pyplot as plt
        # # plt.imshow(proposal[0, :, :].detach().cpu().numpy())
        # plt.imshow(instance_map.detach().cpu().numpy())
        # plt.show()
        # print(' ')
        return instance_map, instances


class Logger:

    def __init__(self, keys, title=""):

        self.data = {k: [] for k in keys}
        self.title = title
        self.win = None

        print('Created logger with keys:  {}'.format(keys))

    def plot(self, save=False, save_dir=""):

        if self.win is None:
            self.win = plt.subplots()
        fig, ax = self.win
        ax.cla()

        keys = []
        count = 0
        for key in self.data:
            if (count < 3):
                keys.append(key)
                data = self.data[key]
                ax.plot(range(len(data)), data, marker='.')
                count += 1
        ax.legend(keys, loc='upper right')
        ax.set_title(self.title)

        plt.draw()
        plt.close(fig)
        # Visualizer.mypause(0.001)

        if save:
            # save figure
            fig.savefig(os.path.join(save_dir, self.title + '.png'))

            # save data as csv
            df = pd.DataFrame.from_dict(self.data)
            df.to_csv(os.path.join(save_dir, self.title + '.csv'))

    def add(self, key, value):
        assert key in self.data, "Key not in data"
        self.data[key].append(value)


def degrid(meter, grid_size, pixel_size):
    return int(meter * (grid_size - 1) / pixel_size + 1)


def add_samples(samples, ax, n, amax):
    samples_list = []
    for i in range(samples.shape[1]):
        samples_list.append(degrid(samples[ax, i], n, amax))
    return samples_list


def prepare_embedding_for_test_image(instance_map, output, grid_x, grid_y, pixel_x, pixel_y, predictions, n_sigma):
    instance_ids = instance_map.unique()
    instance_ids = instance_ids[instance_ids != 0]

    xm = torch.linspace(0, pixel_x, grid_x).view(1, 1, -1).expand(1, grid_y, grid_x)
    ym = torch.linspace(0, pixel_y, grid_y).view(1, -1, 1).expand(1, grid_y, grid_x)
    xym = torch.cat((xm, ym), 0)
    height, width = instance_map.size(0), instance_map.size(1)
    xym_s = xym[:, 0:height, 0:width].contiguous()
    spatial_emb = torch.tanh(output[0, 0:2]).cpu() + xym_s
    sigma = output[0, 2:2 + n_sigma]
    color_sample = sns.color_palette("dark")
    color_embedding = sns.color_palette("bright")
    color_sample_dic = {}
    color_embedding_dic = {}
    samples_x = {}
    samples_y = {}
    sample_spatial_embedding_x = {}
    sample_spatial_embedding_y = {}
    center_x = {}
    center_y = {}
    sigma_x = {}
    sigma_y = {}
    for id in instance_ids:
        in_mask = instance_map.eq(id)
        xy_in = xym_s[in_mask.expand_as(xym_s)].view(2, -1)  # 2 N
        perm = torch.randperm(xy_in.size(1))
        idx = perm[:5]
        samples = xy_in[:, idx]
        samples_x[id.item()] = add_samples(samples, 0, grid_x - 1, pixel_x)
        samples_y[id.item()] = add_samples(samples, 1, grid_y - 1, pixel_y)

        # embeddings
        spatial_emb_in = spatial_emb[in_mask.expand_as(spatial_emb)].view(2, -1)
        samples_spatial_embeddings = spatial_emb_in[:, idx]

        sample_spatial_embedding_x[id.item()] = add_samples(samples_spatial_embeddings, 0, grid_x - 1, pixel_x)
        sample_spatial_embedding_y[id.item()] = add_samples(samples_spatial_embeddings, 1, grid_y - 1, pixel_y)
        center_image = predictions[id.item() - 1]['center-image']  # predictions is a list!
        center_mask = in_mask & center_image.byte()

        if (center_mask.sum().eq(1)):
            center = xym_s[center_mask.expand_as(xym_s)].view(2, 1, 1)
        else:
            xy_in = xym_s[in_mask.expand_as(xym_s)].view(2, -1)
            center = xy_in.mean(1).view(2, 1, 1)  # 2 x 1 x 1

        center_x[id.item()] = degrid(center[0], grid_x - 1, pixel_x)
        center_y[id.item()] = degrid(center[1], grid_y - 1, pixel_y)

        # sigma
        s = sigma[in_mask.expand_as(sigma)].view(n_sigma, -1).mean(1)
        s = torch.exp(s * 10)
        sigma_x_tmp = 0.5 / s[0]
        sigma_y_tmp = 0.5 / s[1]
        sigma_x[id.item()] = degrid(torch.sqrt(sigma_x_tmp), grid_x - 1, pixel_x)
        sigma_y[id.item()] = degrid(torch.sqrt(sigma_y_tmp), grid_y - 1, pixel_y)

        # colors
        color_sample_dic[id.item()] = color_sample[int(id % 10)]
        color_embedding_dic[id.item()] = color_embedding[int(id % 10)]

    return center_x, center_y, samples_x, samples_y, sample_spatial_embedding_x, \
           sample_spatial_embedding_y, sigma_x, sigma_y, color_sample_dic, color_embedding_dic
