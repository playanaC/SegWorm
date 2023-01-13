import ast
import os
from glob import glob
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pycocotools.mask as rletools
import tifffile
from scipy.ndimage import zoom
from skimage.segmentation import relabel_sequential
from EmbedSeg_files.utils import invert_one_hot


def visualize0(image, prediction, new_cmp):
    """
        Visualizes 2 x 2 grid with Top-Left (Image), Top-Right (Ground Truth), Bottom-Left (Seed),
        Bottom-Right (Instance Segmentation Prediction)

        Parameters
        -------

        image: Numpy Array (YX or 1YX)
            Raw Image
        prediction: Numpy Array (YX)
            Model Prediction of Instance Segmentation
        new_cmp: Color Map
        Returns
        -------
        """

    font = {'family': 'serif',
            'color': 'white',
            'weight': 'bold',
            'size': 16,
            }
    prediction_color = invert_one_hot(prediction)
    plt.figure(figsize=(15, 15))
    img_show = image if image.ndim == 2 else image[0, ...]
    plt.subplot(121)
    plt.imshow(img_show, cmap='magma');
    plt.text(30, 30, "IM", fontdict=font)
    plt.xlabel('Image')
    plt.axis('off')
    plt.subplot(122)
    plt.axis('off')
    plt.imshow(prediction_color, cmap=new_cmp, interpolation='None')
    plt.text(30, 30, "PRED", fontdict=font)
    for i in range(prediction.shape[0]):
        img00 = prediction[i, :, :]
        mass_x, mass_y = np.where(img00 > 0)
        cent_x = np.average(mass_x)
        cent_y = np.average(mass_y)
        plt.text(cent_y, cent_x, i, fontdict=font, bbox=dict(facecolor='black', edgecolor='red', linewidth=2))
    plt.xlabel('Prediction')
    plt.tight_layout()


def visualize2(image, results_masks, new_cmp):
    """
        Visualizes 2 x 2 grid with Top-Left (Image), Top-Right (Ground Truth), Bottom-Left (Seed),
        Bottom-Right (Instance Segmentation Prediction)

        Parameters
        -------
        image: Raw image
        results_masks: Dictionary (good_masks, bad_masks, good_index, bad_index)
            Image
        new_cmp: Color Map
        Returns
        -------
        """

    font = {'family': 'serif',
            'color': 'white',
            'weight': 'bold',
            'size': 16,
            }

    worms_true = results_masks['worms_true']
    worms_bads = results_masks['worms_bads']
    index_good = results_masks['index_good']
    index_bad = results_masks['index_bad']

    plt.figure(figsize=(15, 15))
    plt.subplot(121)
    plt.text(30, 30, "GOOD masks", fontdict=font)
    try:
        worms_true_color = invert_one_hot(worms_true)
        plt.imshow(worms_true_color, cmap=new_cmp, interpolation='None')
        for i in range(worms_true.shape[0]):
            img00 = worms_true[i, :, :]
            mass_x, mass_y = np.where(img00 > 0)
            cent_x = np.average(mass_x)
            cent_y = np.average(mass_y)
            plt.text(cent_y, cent_x, index_good[i], fontdict=font, bbox=dict(facecolor='black', edgecolor='red', linewidth=2))
        # plt.xlabel('Prediction')
        plt.tight_layout()
    except:
        worms_true_color = np.zeros((image.shape[0], image.shape[1]), np.int8)
        plt.imshow(worms_true_color, interpolation='None')
    plt.axis('off')

    plt.subplot(122)
    plt.text(30, 30, "BAD masks", fontdict=font)
    try:
        worms_bads_color = invert_one_hot(worms_bads)
        plt.imshow(worms_bads_color, cmap=new_cmp, interpolation='None')
        for i in range(worms_bads.shape[0]):
            img00 = worms_bads[i, :, :]
            mass_x, mass_y = np.where(img00 > 0)
            cent_x = np.average(mass_x)
            cent_y = np.average(mass_y)
            plt.text(cent_y, cent_x, index_bad[i], fontdict=font,
                     bbox=dict(facecolor='black', edgecolor='red', linewidth=2))
        # plt.xlabel('Prediction')
        plt.tight_layout()
    except:
        worms_bads_color = np.zeros((image.shape[0], image.shape[1]), np.int8)
        plt.imshow(worms_bads_color, interpolation='None')


def visualize_post0(image, prediction, results_masks, new_cmp, save_img_results):
    font = {'family': 'serif',
            'color': 'white',
            'weight': 'bold',
            'size': 16,
            }

    worms_true = results_masks['worms_true']
    worms_bads = results_masks['worms_bads']
    index_good = results_masks['index_good']
    index_bad = results_masks['index_bad']

    plt.ioff()  # Turn interactive plotting off
    plt.figure(figsize=(15, 15))

    img_show = image if image.ndim == 2 else image[0, ...]
    plt.subplot(221)
    plt.axis('off')
    plt.imshow(img_show, cmap='magma')
    plt.text(30, 30, "IM", fontdict=font)
    plt.tight_layout()
    plt.xlabel('Image')

    plt.subplot(223)
    plt.axis('off')
    plt.text(30, 30, "GOOD masks", fontdict=font)
    try:
        worms_true_color = invert_one_hot(worms_true)
        plt.imshow(worms_true_color, cmap=new_cmp, interpolation='None')
        for i in range(worms_true.shape[0]):
            img00 = worms_true[i, :, :]
            mass_x, mass_y = np.where(img00 > 0)
            cent_x = np.average(mass_x)
            cent_y = np.average(mass_y)
            plt.text(cent_y, cent_x, index_good[i], fontdict=font,
                     bbox=dict(facecolor='black', edgecolor='red', linewidth=2))
        # plt.xlabel('Prediction')
        # plt.tight_layout()
    except:
        worms_true_color = np.zeros((image.shape[0], image.shape[1]), np.int8)
        plt.imshow(worms_true_color, interpolation='None')
    plt.tight_layout()

    plt.subplot(224)
    plt.axis('off')
    plt.text(30, 30, "BAD masks", fontdict=font)
    try:
        worms_bads_color = invert_one_hot(worms_bads)
        plt.imshow(worms_bads_color, cmap=new_cmp, interpolation='None')
        for i in range(worms_bads.shape[0]):
            img00 = worms_bads[i, :, :]
            mass_x, mass_y = np.where(img00 > 0)
            cent_x = np.average(mass_x)
            cent_y = np.average(mass_y)
            plt.text(cent_y, cent_x, index_bad[i], fontdict=font,
                     bbox=dict(facecolor='black', edgecolor='red', linewidth=2))
    except:
        worms_bads_color = np.zeros((image.shape[0], image.shape[1]), np.int8)
        plt.imshow(worms_bads_color, interpolation='None')
    plt.tight_layout()

    # plt.subplot(222)
    # plt.axis('off')
    # plt.text(30, 30, "PRED", fontdict=font)
    # try:
    #     prediction_color = invert_one_hot(prediction)
    #     plt.imshow(prediction_color, cmap=new_cmp, interpolation='None')
    # except:
    #     prediction_color = np.zeros((image.shape[0], image.shape[1]), np.int8)
    #     plt.imshow(prediction_color, interpolation='None')
    # plt.xlabel('PRED')
    # plt.tight_layout()

    plt.savefig(save_img_results)
    plt.close()


def visualize_post(image, ground_truth, prediction, post_processing, new_cmp, save_img_results):
    font = {'family': 'serif',
            'color': 'white',
            'weight': 'bold',
            'size': 16,
            }

    plt.ioff()  # Turn interactive plotting off
    plt.figure(figsize=(15, 15))
    img_show = image if image.ndim == 2 else image[0, ...]
    plt.subplot(221)
    plt.imshow(img_show, cmap='magma')
    plt.text(30, 30, "IM", fontdict=font)
    plt.xlabel('Image')
    plt.axis('off')
    if ground_truth is not None:
        plt.subplot(222)
        plt.axis('off')
        plt.imshow(ground_truth, cmap=new_cmp, interpolation='None')
        plt.text(30, 30, "GT", fontdict=font)
        plt.xlabel('Ground Truth')
    plt.subplot(223)
    plt.axis('off')
    # plt.imshow(seed, interpolation='None')
    plt.imshow(prediction, cmap=new_cmp, interpolation='None')
    plt.text(30, 30, "PRED", fontdict=font)
    plt.xlabel('PRED')
    plt.subplot(224)
    plt.axis('off')
    plt.imshow(post_processing, cmap=new_cmp, interpolation='None')
    plt.text(30, 30, "POST", fontdict=font)
    plt.xlabel('POST')
    plt.tight_layout()
    plt.savefig(save_img_results)
    # plt.show()
    # plt.savefig(save_img_results)
    plt.close()