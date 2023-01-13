import os
from PIL import Image
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.cm as cm
from EmbedSeg_files.Dataset_EmbedSeg import TwoDimensionalDataset
from EmbedSeg_files.model.BranchedERFNet import BranchedERFNet
from EmbedSeg_files.utils import apply_tta_2d
from EmbedSeg_files.utils import Cluster
import scipy.misc
import matplotlib.pyplot as plt
torch.backends.cudnn.benchmark = True
import numpy as np
from tifffile import imsave
from scipy.ndimage import zoom
from scipy.optimize import minimize_scalar, linear_sum_assignment
from skimage.segmentation import relabel_sequential


def begin_evaluating(test_configs, optimize=False, maxiter=10, verbose=False, mask_region=None):
    """Entry function for inferring on test images

        Parameters
        ----------
        test_configs : dictionary
            Dictionary containing testing-specific parameters (for e.g. the `seed_thresh`  to use)
        optimize : bool, optional
            It is possible to determine the best performing `fg_thresh` by optimizing over different values on the validation sub-set
            By default and in the absence of optimization (i.e. `optimize=False`), the fg_thresh  is set equal to 0.5
        maxiter: int
            Number of iterations of optimization.
            Comes into play, only if `optimize=True`
        verbose: bool, optional
            If set equal to True, prints the AP_dsb for each image individually
        mask_region: list of lists, optional
            If a certain region of the image is not labelled in the GT label mask, that can be specified here.
            This enables comparison of the model prediction only with the area which is labeled in the GT label mask
        Returns
        -------
        result_dic: Dictionary
            Keys include the employed `fg_thresh` and the corresponding `AP_dsb` at IoU threshold = 0.5
        """
    n_sigma = test_configs['n_sigma']
    ap_val = test_configs['ap_val']
    min_mask_sum = test_configs['min_mask_sum']
    min_unclustered_sum = test_configs['min_unclustered_sum']
    min_object_size = test_configs['min_object_size']
    mean_object_size = test_configs['mean_object_size']
    tta = test_configs['tta']
    seed_thresh = test_configs['seed_thresh']
    fg_thresh = test_configs['fg_thresh']
    save_images = test_configs['save_images']
    save_results = test_configs['save_results']
    save_dir = test_configs['save_dir']
    anisotropy_factor = test_configs['anisotropy_factor']
    grid_x = test_configs['grid_x']
    grid_y = test_configs['grid_y']
    grid_z = test_configs['grid_z']
    pixel_x = test_configs['pixel_x']
    pixel_y = test_configs['pixel_y']
    pixel_z = test_configs['pixel_z']
    one_hot = test_configs['dataset']['kwargs']['one_hot']
    cluster_fast = test_configs['cluster_fast']
    expand_grid = test_configs['expand_grid']
    uniform_ds_factor = test_configs['uniform_ds_factor']

    # set device
    device = torch.device("cuda:0" if test_configs['cuda'] else "cpu")

    # dataloader
    dataset = TwoDimensionalDataset(**test_configs['dataset']['kwargs'])
    dataset_it = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=test_configs['num_workers'],
                                             pin_memory=True if torch.cuda.is_available() else False)

    # load model
    model = BranchedERFNet(**test_configs['model']['kwargs'])
    model = torch.nn.DataParallel(model).to(device)

    # load snapshot
    if os.path.exists(test_configs['checkpoint_path']):
        state = torch.load(test_configs['checkpoint_path'], map_location=device)
        model.load_state_dict(state['model_state_dict'], strict=True)
    else:
        assert False, 'checkpoint_path {} does not exist!'.format(test_configs['checkpoint_path'])

    # test on evaluation images:
    result_dic = {}
    if test_configs['name'] == '2d':
        args = (seed_thresh, ap_val, min_mask_sum, min_unclustered_sum, min_object_size, tta,
                model, dataset_it, save_images, save_results, save_dir, verbose, grid_x,
                grid_y, pixel_x, pixel_y, one_hot, n_sigma, cluster_fast, expand_grid)

        result = test(fg_thresh, *args)
        result_dic['fg_thresh'] = fg_thresh
        result_dic['AP_dsb_05'] = -result

    return result_dic


def predict(im, model, tta, cluster_fast, n_sigma, fg_thresh, seed_thresh, min_mask_sum, min_unclustered_sum,
            min_object_size,
            cluster):
    """

    Parameters
    ----------
    im : PyTorch Tensor
        BCYX

    model: PyTorch model

    tta: bool
        If True, then Test-Time Augmentation is on, otherwise off
    cluster_fast: bool
        If True, then the cluster.cluster() is used
        If False, then cluster.cluster_local_maxima() is used
    n_sigma: int
        This should be set equal to `2` for a 2D setting
    fg_thresh: float
        This should be set equal to `0.5` by default
    seed_thresh: float
        This should be set equal to `0.9` by default
    min_mask_sum: int
        Only start creating instances, if there are at least `min_mask_sum` pixels in foreground!
    min_unclustered_sum: int
        Stop when the number of seed candidates are less than `min_unclustered_sum`
    min_object_size: int
        Predicted Objects below this threshold are ignored

    cluster: Object of class `Cluster`

    Returns
    -------
    instance_map: PyTorch Tensor
        YX
    seed_map: PyTorch Tensor
        YX
    """

    multiple_y = im.shape[2] // 8
    multiple_x = im.shape[3] // 8

    if im.shape[2] % 8 != 0:
        diff_y = 8 * (multiple_y + 1) - im.shape[2]
    else:
        diff_y = 0
    if im.shape[3] % 8 != 0:
        diff_x = 8 * (multiple_x + 1) - im.shape[3]
    else:
        diff_x = 0
    p2d = (diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2)  # last dim, second last dim

    im = F.pad(im, p2d, "reflect")

    if (tta):
        output = apply_tta_2d(im, model)
    else:
        output = model(im)

    # print(output.size())
    # import matplotlib.pyplot as plt
    # plt.imshow(output[0,4,:,:].detach().cpu().numpy())
    # plt.show()

    if cluster_fast:
        instance_map, instances_all = cluster.cluster(output[0],
                                                      n_sigma=n_sigma,
                                                      seed_thresh=seed_thresh,
                                                      min_mask_sum=min_mask_sum,
                                                      min_unclustered_sum=min_unclustered_sum,
                                                      min_object_size=min_object_size)
    else:
        instance_map = cluster.cluster_local_maxima(output[0],
                                                    n_sigma=n_sigma,
                                                    fg_thresh=fg_thresh,
                                                    min_mask_sum=min_mask_sum,
                                                    min_unclustered_sum=min_unclustered_sum,
                                                    min_object_size=min_object_size)
    # print('instances_all', len(instances_all))
    seed_map = torch.sigmoid(output[0, -1, ...])
    # unpad instance_map, seed_map
    if (diff_y - diff_y // 2) is not 0:
        instance_map = instance_map[diff_y // 2:-(diff_y - diff_y // 2), ...]
        seed_map = seed_map[diff_y // 2:-(diff_y - diff_y // 2), ...]
    if (diff_x - diff_x // 2) is not 0:
        instance_map = instance_map[..., diff_x // 2:-(diff_x - diff_x // 2)]
        seed_map = seed_map[..., diff_x // 2:-(diff_x - diff_x // 2)]
    return instance_map, seed_map, instances_all


def test(fg_thresh, *args):
    """Infer the trained 2D model on 2D images

            Parameters
            ----------
            fg_thresh : float
                foreground threshold decides which pixels are considered for clustering, based on the predicted seediness scores at these pixels.
            args: dictionary
                Contains other paremeters such as `ap_val`, `seed_thresh` etc
            Returns
            -------
            float
                Average `AP_dsb` over all test images
            """
    seed_thresh, ap_val, min_mask_sum, min_unclustered_sum, min_object_size, tta, model, dataset_it, save_images, \
    save_results, save_dir, verbose, grid_x, grid_y, pixel_x, pixel_y, one_hot, n_sigma, cluster_fast, expand_grid = args

    model.eval()

    # cluster module
    cluster = Cluster(grid_y, grid_x, pixel_y, pixel_x)

    with torch.no_grad():
        result_list = []
        image_file_names = []
        for sample in tqdm(dataset_it):

            im = sample['image']  # B 1 Y X
            H, W = im.shape[2], im.shape[3]
            if H > grid_y or W > grid_x:
                if expand_grid:
                    # simple trick to expand the grid while keeping pixel resolution the same as before
                    H_, W_ = round_up_8(H), round_up_8(W)
                    temp = np.maximum(H_, W_)
                    H_ = temp
                    W_ = temp
                    pixel_x_modified = pixel_y_modified = H_ / grid_y
                    cluster = Cluster(H_, W_, pixel_y_modified, pixel_x_modified)
                    instance_map, seed_map = predict(im, model, tta, cluster_fast,
                                                     n_sigma, fg_thresh, seed_thresh, min_mask_sum, min_unclustered_sum,
                                                     min_object_size, cluster)

                else:
                    # here, we try stitching predictions instead
                    last = 1
                    instance_map = np.zeros((H, W), dtype=np.int16)
                    seed_map = np.zeros((H, W), dtype=np.float)
                    num_overlap_pixels = 4
                    for y in range(0, H, grid_y - num_overlap_pixels):
                        for x in range(0, W, grid_x - num_overlap_pixels):
                            instance_map_tile, seed_map_tile = predict(im[:, :, y:y + grid_y, x:x + grid_x], model, tta,
                                                                       cluster_fast,
                                                                       n_sigma, fg_thresh, seed_thresh, min_mask_sum,
                                                                       min_unclustered_sum, min_object_size,
                                                                       cluster)
                            last, instance_map = stitch_2d(instance_map_tile.cpu().detach().numpy(), instance_map, y, x,
                                                           last, num_overlap_pixels)
                            seed_map[y:y + grid_y, x:x + grid_x] = seed_map_tile.cpu().detach().numpy()
                    instance_map = torch.from_numpy(instance_map).cuda()
                    seed_map = torch.from_numpy(seed_map).float().cuda()
            else:
                instance_map, seed_map, instace_all = predict(im, model, tta, cluster_fast,
                                                 n_sigma, fg_thresh, seed_thresh, min_mask_sum, min_unclustered_sum,
                                                 min_object_size, cluster)

            base, _ = os.path.splitext(os.path.basename(sample['im_name'][0]))
            image_file_names.append(base)
            # print('instace_all', len(instace_all))

            # if (one_hot):
            #     if ('instance' in sample):
            #         all_results = obtain_APdsb_one_hot(gt_image=sample['instance'].squeeze().cpu().detach().numpy(),
            #                                            prediction_image=instance_map.cpu().detach().numpy(),
            #                                            ap_val=ap_val)
            #         if (verbose):
            #             print("Accuracy: {:.03f}".format(all_results), flush=True)
            #         result_list.append(all_results)
            # else:
            #     if ('instance' in sample):
            #         all_results = matching_dataset(y_true=[sample['instance'].squeeze().cpu().detach().numpy()],
            #                                        y_pred=[instance_map.cpu().detach().numpy()], thresh=ap_val,
            #                                        show_progress=False)
            #         if (verbose):
            #             print("Accuracy: {:.03f}".format(all_results.accuracy), flush=True)
            #         result_list.append(all_results.accuracy)

            if save_images and ap_val == 0.5:

                if not os.path.exists(os.path.join(save_dir, 'predictions/')):
                    os.makedirs(os.path.join(save_dir, 'predictions/'))
                    print("Created new directory {}".format(os.path.join(save_dir, 'predictions/')))
                if not os.path.exists(os.path.join(save_dir, 'ground-truth/')):
                    os.makedirs(os.path.join(save_dir, 'ground-truth/'))
                    print("Created new directory {}".format(os.path.join(save_dir, 'ground-truth/')))
                if not os.path.exists(os.path.join(save_dir, 'seeds/')):
                    os.makedirs(os.path.join(save_dir, 'seeds/'))
                    print("Created new directory {}".format(os.path.join(save_dir, 'seeds/')))

                # save predictions
                base, _ = os.path.splitext(os.path.basename(sample['im_name'][0]))
                instances_file = os.path.join(save_dir, 'predictions/', base + '.tif')
                imsave(instances_file, instance_map.cpu().detach().numpy().astype(
                    np.uint16))
                seeds_file = os.path.join(save_dir, 'seeds/', base + '.tif')
                imsave(seeds_file, seed_map.cpu().detach().numpy())
                if 'instance' in sample:
                    gt_file = os.path.join(save_dir, 'ground-truth/', base + '.tif')
                    imsave(gt_file, sample['instance'].squeeze().cpu().detach().numpy())

                # # save masks separately
                # if not os.path.exists(os.path.join(save_dir, 'predictions_individual_masks/')):
                #     os.makedirs(os.path.join(save_dir, 'predictions_individual_masks/'))
                #     print("Created new directory {}".format(os.path.join(save_dir, 'predictions_individual_masks/')))
                # if not os.path.exists(os.path.join(save_dir, 'predictions_individual_masks/' + base + '/')):
                #     os.makedirs(os.path.join(save_dir, 'predictions_individual_masks/' + base + '/'))
                # for mki in range(len(instace_all)):
                #     name_mask_i = save_dir + '/predictions_individual_masks/' + base + '/' + str(mki) + '.png'
                #     plt.imsave(name_mask_i, instace_all[mki], cmap=cm.gray)

                # save mask in TIF format (matrix)
                if not os.path.exists(os.path.join(save_dir, 'predictions_masks/')):
                    os.makedirs(os.path.join(save_dir, 'predictions_masks/'))
                    print("Created new directory {}".format(os.path.join(save_dir, 'predictions_masks/')))

                # converting list to array
                name_mask_list = save_dir + '/predictions_masks/' + base + '.tif'
                instace_list = np.array(instace_all)
                imsave(name_mask_list, instace_list)

        if save_results and 'instance' in sample:
            if not os.path.exists(os.path.join(save_dir, 'results/')):
                os.makedirs(os.path.join(save_dir, 'results/'))
                print("Created new directory {}".format(os.path.join(save_dir, 'results/')))
            txt_file = os.path.join(save_dir,
                                    'results/combined_AP-' + '{:.02f}'.format(ap_val) + '_tta-' + str(tta) + '.txt')
            with open(txt_file, 'w') as f:
                f.writelines(
                    "image_file_name, min_mask_sum, min_unclustered_sum, min_object_size, seed_thresh, intersection_threshold, accuracy \n")
                f.writelines("+++++++++++++++++++++++++++++++++\n")
                for ind, im_name in enumerate(image_file_names):
                    im_name_tif = im_name + '.tif'
                    score = result_list[ind]
                    f.writelines(
                        "{} {:.02f} {:.02f} {:.02f} {:.02f} {:.02f} {:.05f} \n".format(im_name_tif, min_mask_sum,
                                                                                       min_unclustered_sum,
                                                                                       min_object_size, seed_thresh,
                                                                                       ap_val, score))
                f.writelines("+++++++++++++++++++++++++++++++++\n")
                f.writelines("Average Precision (AP_dsb)  {:.02f} {:.05f}\n".format(ap_val, np.mean(result_list)))

        if len(result_list) != 0:
            print(
                "Mean Average Precision (AP_dsb) at IOU threshold = {} at foreground threshold = {:.05f}, is equal to {:.05f}".format(
                    ap_val, fg_thresh, np.mean(result_list)))
            return -np.mean(result_list)
        else:
            return 0.0


def round_up_8(x):
    """Helper function for rounding integer to next multiple of 8

        e.g:
        round_up_8(10) = 16

            Parameters
            ----------
            x : int
                Integer
            Returns
            -------
            int
            """
    return (int(x) + 7) & (-8)


def stitch_2d(instance_map_tile, instance_map_current, y_current=None, x_current=None, last=1, num_overlap_pixels=4):
    """
    Stitching instance segmentations together in case the full 2D image doesn't fit in one go, on the GPU
    This function is executed only if `expand_grid` is set to False
    The key idea is we identify the unique ids in the instance_map_current and the tile, but only in the overlap region.
    Then we look at the IoU of these. If there is more than 50 % IoU, then these are considered to be the same
    Else, a new id is generated!

    Parameters
    ----------
    instance_map_tile : numpy array
        instance segmentation over a tiled view of the image

    instance_map_current: numpy array
        instance segmentation over the complete, large image


    y_current: int
        y position of the top left corner of the tile wrt the complete image

    x_current: int
        x position of the top left corner of the tile wrt the complete image

    last: int
        number of objects currently present in the `instance_map_current`

    num_overlap_pixels: int
        number of overlapping pixels while considering the next tile

    Returns
    -------
    tuple (int, numpy array)
        (updated number of objects currently present in the `instance_map_current`,
        updated instance segmentation over the full image)

    """
    mask = instance_map_tile > 0  # foreground region, which has been clustered

    h_tile = instance_map_tile.shape[0]
    w_tile = instance_map_tile.shape[1]
    h_current = instance_map_current.shape[0]
    w_current = instance_map_current.shape[1]

    instance_map_tile_sequential = np.zeros_like(instance_map_tile)

    if mask.sum() > 0:  # i.e. there were some object predictions
        # make sure that instance_map_tile is labeled sequentially

        ids, _, _ = relabel_sequential(instance_map_tile[mask])
        instance_map_tile_sequential[mask] = ids
        instance_map_tile = instance_map_tile_sequential

        # next pad the tile so that it is aligned wrt the complete image
        # note that doing the padding ensures that the instance_map_tile is the same size as the instance_map_current

        instance_map_tile = np.pad(instance_map_tile,
                                   ((y_current, np.maximum(0, h_current - y_current - h_tile)),
                                    (x_current, np.maximum(0, w_current - x_current - w_tile))))

        # ensure that it has the same shape as instance_map_current
        instance_map_tile = instance_map_tile[:h_current, :w_current]

        mask_overlap = np.zeros_like(
            instance_map_tile)  # this just identifies the region where the tile overlaps with the `instance_map_current`

        if y_current == 0 and x_current == 0:
            ids_tile = np.unique(instance_map_tile)
            ids_tile = ids_tile[ids_tile != 0]  # ignore background
            instance_map_current[:h_current, :w_current] = instance_map_tile
            last = len(ids_tile) + 1
        else:
            if x_current != 0 and y_current == 0:
                mask_overlap[y_current:y_current + h_tile, x_current:x_current + num_overlap_pixels] = 1
            elif x_current == 0 and y_current != 0:
                mask_overlap[y_current:y_current + num_overlap_pixels, x_current:x_current + w_tile] = 1
            elif x_current != 0 and y_current != 0:
                mask_overlap[y_current:y_current + h_tile, x_current:x_current + num_overlap_pixels] = 1
                mask_overlap[y_current:y_current + num_overlap_pixels, x_current:x_current + w_tile] = 1

            # identify ids in the complete tile, not just the overlap region,
            ids_tile_all = np.unique(instance_map_tile)
            ids_tile_all = ids_tile_all[ids_tile_all != 0]

            # identify ids in the the overlap region,
            ids_tile_overlap = np.unique(instance_map_tile * mask_overlap)
            ids_tile_overlap = ids_tile_overlap[ids_tile_overlap != 0]

            # identify ids not in overlap region
            ids_tile_notin_overlap = np.setdiff1d(ids_tile_all, ids_tile_overlap)

            # identify ids in `instance_map_current` but only in the overlap region
            instance_map_current_masked = torch.from_numpy(instance_map_current * mask_overlap).cuda()

            ids_current_overlap = torch.unique(instance_map_current_masked).cpu().detach().numpy()
            ids_current_overlap = ids_current_overlap[ids_current_overlap != 0]

            IoU_table = np.zeros((len(ids_tile_overlap), len(ids_current_overlap)))
            instance_map_tile_masked = torch.from_numpy(instance_map_tile * mask_overlap).cuda()

            # rows are ids in tile, cols are ids in GT instance map

            for i, id_tile_overlap in enumerate(ids_tile_overlap):
                for j, id_current_overlap in enumerate(ids_current_overlap):

                    intersection = ((instance_map_tile_masked == id_tile_overlap)
                                    & (instance_map_current_masked == id_current_overlap)).sum()
                    union = ((instance_map_tile_masked == id_tile_overlap)
                             | (instance_map_current_masked == id_current_overlap)).sum()
                    if union != 0:
                        IoU_table[i, j] = intersection / union
                    else:
                        IoU_table[i, j] = 0.0

            row_indices, col_indices = linear_sum_assignment(-IoU_table)
            matched_indices = np.array(list(zip(row_indices, col_indices)))  # list of (row, col) tuples
            unmatched_indices_tile = np.setdiff1d(np.arange(len(ids_tile_overlap)), row_indices)

            for m in matched_indices:
                if (IoU_table[m[0], m[1]] >= 0.5):  # (tile, current)
                    # wherever the tile is m[0], it should be assigned m[1] in the larger prediction image
                    instance_map_current[instance_map_tile == ids_tile_overlap[m[0]]] = ids_current_overlap[m[1]]
                elif (IoU_table[m[0], m[1]] == 0):
                    # there is no intersection
                    instance_map_current[instance_map_tile == ids_tile_overlap[m[0]]] = last
                    last += 1
                else:
                    # otherwise just take a union of the both ...
                    # basically this should spawn a new label, since there was not a satisfactory match with any pre-existing id in the instance_map_current
                    instance_map_current[instance_map_tile == ids_tile_overlap[m[0]]] = last
                    # instance_map_current[instance_map_current == ids_current_overlap[m[1]]] = last  # TODO
                    last += 1
            for index in unmatched_indices_tile:  # not a tuple
                instance_map_current[instance_map_tile == ids_tile_overlap[index]] = last
                last += 1
            for id in ids_tile_notin_overlap:
                instance_map_current[instance_map_tile == id] = last
                last += 1
        return last, instance_map_current
    else:
        return last, instance_map_current  # if there are no ids in tile, then just return