from tqdm import tqdm
from glob import glob
import tifffile
import numpy as np
import os
import cv2
import math
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from skimage import measure, morphology
from scipy import ndimage


def check_inside_folder(path_folder):
    folder_files = os.listdir(path_folder + '/')
    if len(folder_files)>0:
        CHK = True
    else:
        CHK = False
    return CHK


def check_folder(folder_main):
    # check if folders exist
    path_final = ''
    if os.path.exists(folder_main):
        CHK = check_inside_folder(folder_main)
        path_final = folder_main + '/'
    else:
        CHK = False

    return CHK, path_final


def chk_file_extention(file):
    CHK = False
    if file.endswith('.tif'):
        CHK = True
    elif file.endswith('.tiff'):
        CHK = True
    elif file.endswith('.TIF'):
        CHK = True
    elif file.endswith('.TIFF'):
        CHK = True
    return CHK


def get_image_folder(size, path_folder):
    list_folders = os.listdir(path_folder)
    list_imges = []
    for i in range(len(list_folders)):
        if chk_file_extention(list_folders[i]):
            list_imges.append(list_folders[i])

    masks = np.zeros((len(list_imges),size[0], size[1]), np.int8)
    for i in range(len(list_imges)):
        path_mask = path_folder + list_imges[i]
        img_tiff = tifffile.imread(path_mask)
        img_tiff = ((img_tiff < 250) * 1).astype('uint8')
        masks[i, :, :] = img_tiff
        # plt.imshow(image)
        # plt.show()
    return masks


def get_files(path_file, path_masks, path_obscuredmasks):
    image = tifffile.imread(path_file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    size = gray.shape

    path_masks_exist = os.path.exists(path_masks)
    if path_masks_exist:
        masks = get_image_folder(size, path_masks)
    else:
        masks = np.zeros((0, size[0], size[1]), np.int8)

    path_obscuredmasks_exist = os.path.exists(path_obscuredmasks)
    if path_obscuredmasks_exist:
        masks_obscured = get_image_folder(size, path_obscuredmasks)
    else:
        masks_obscured = np.zeros((0, size[0], size[1]), np.int8)

    all_mask = np.concatenate((masks, masks_obscured), axis=0)

    return gray, all_mask


def branchs_endpoints(img):
    # Find row and column locations that are non-zero
    (rows, cols) = np.nonzero(img)

    # Initialize empty list of co-ordinates branch
    skel_coords_branch = []

    # Initialize empty list of co-ordinates endpoints
    skel_coords_endpoints = []

    # For each non-zero pixel...
    for (r, c) in zip(rows, cols):

        # Extract an 8-connected neighbourhood
        (col_neigh, row_neigh) = np.meshgrid(np.array([c - 1, c, c + 1]), np.array([r - 1, r, r + 1]))

        # Cast to int to index into image
        col_neigh = col_neigh.astype('int')
        row_neigh = row_neigh.astype('int')

        # Convert into a single 1D array and check for non-zero locations
        pix_neighbourhood = img[row_neigh, col_neigh].ravel() != 0

        # If the number of non-zero locations equals 2, add this to
        # our list of co-ordinates
        if np.sum(pix_neighbourhood) > 3:
            skel_coords_branch.append((r, c))

        # If the number of non-zero locations equals 2, add this to
        # our list of co-ordinates
        if np.sum(pix_neighbourhood) == 2:
            skel_coords_endpoints.append((r, c))

    return skel_coords_branch, skel_coords_endpoints


def update_worms(worms_true, masks):
    masks = np.expand_dims(masks, axis=0)
    if worms_true.max() > 0:
        worms_true = np.concatenate((worms_true, masks), axis=0)
    else:
        worms_true = masks

    return worms_true


def edge_img(img):
    # create edge
    masK_edge = np.zeros((img.shape[0], img.shape[1]), dtype="uint8")
    masK_edge[0:img.shape[0], 0:1] = 255
    masK_edge[0:img.shape[0], img.shape[1] - 1:img.shape[1]] = 255

    masK_edge[0:1, 0:img.shape[1]] = 255
    masK_edge[img.shape[0] - 1:img.shape[0], 0:img.shape[1]] = 255
    return masK_edge


def edge_worm(img):
    masK_edge = edge_img(img)
    # check edge worm
    masK_worms = cv2.bitwise_and((img*255).astype('uint8'), masK_edge)
    worm_edge = np.count_nonzero(masK_worms)
    # plt.imshow(masK_worms)
    # plt.show()

    CHK = 0
    if worm_edge > 0:
        CHK = 1
    return CHK


def true_worm(img):

    labels = measure.label(img.astype('uint8'), background=0)
    if labels.max() > 1:
        size_parts = []
        for i in range(1, labels.max()+1):
            size_bw = np.count_nonzero(((labels == i)*1))
            size_parts.append(size_bw)
        max_bw = size_parts.index(max(size_parts)) + 1
        worm = (labels == max_bw) * 1
    else:
        worm = img

        # print(labels.max())
    #
    # plt.imshow(((labels == 2)*255).astype(np.uint8))
    # plt.show()
    # plt.imshow(img)
    # plt.show()

    return worm


def check_skl(skeleton, branchs):

    brach_mask = np.zeros((skeleton.shape[0], skeleton.shape[1]), dtype="uint8")
    for i in range(len(branchs)):
        # Extract an 8-connected neighbourhood
        r = branchs[i][0]
        c = branchs[i][1]
        brach_mask[r, c] = 255
    skeleton_BM = cv2.bitwise_and(skeleton.astype('uint8'), cv2.bitwise_not(brach_mask))
    labels = measure.label(skeleton_BM.astype('uint8'), background=0)

    skl_size = np.count_nonzero(skeleton > 0)
    size_parts = []
    cnt = 0
    for i in range(1, labels.max() + 1):
        part_i = np.count_nonzero((labels == i)*255 > 0) / skl_size
        size_parts.append(part_i)
        if part_i > 0.6:
            cnt = cnt + 1

    return cnt


def next_8C(point_XY, skeleton):  # Extract an 8-connected neighbourhood
    r = point_XY[0]
    c = point_XY[1]
    C8 = []
    for i in range(c - 1, c + 2):
        for j in range(r - 1, r + 2):
            if not(i == c and j == r):
                if skeleton[j, i] == 1:
                    C8.append([j, i])
    return C8


def get_allSK(skeleton_i, P1, PF, SKLS):
    P2 = P1
    temp_PF = []
    while len(P2) != 0:
        P2 = next_8C(P1[0], skeleton_i)
        PF.append(P1[0])
        skeleton_i[P1[0][0], P1[0][1]] = 0
        if len(P2) != 0:
            if len(P2) == 1:
                P1 = P2
            else:
                for i in range(len(P2)):
                    P3 = P2[i]
                    temp = np.copy(skeleton_i)
                    temp_PF = PF.copy()
                    SKLS = get_allSK(temp, [P3], temp_PF, SKLS)
                P2 = []
    if len(temp_PF) == 0:
        SKLS.append(PF)
    return SKLS


def build_skel(H, W, XY_skels):
    mask = np.zeros((H, W))
    for XY_skel in XY_skels:
        mask[XY_skel[0], XY_skel[1]] = 1
    return mask


def iou_img(I1, I2):
    IoU_dec = 0
    I1 = I1.astype(np.uint8)
    I2 = I2.astype(np.uint8)
    # f, (ax0, ax1) = plt.subplots(1, 2)
    # ax0.imshow(img_dec)
    # ax1.imshow(true_masks)
    # plt.show()

    AND = cv2.bitwise_and(I1, I2)
    OR = cv2.bitwise_or(I1, I2)
    C0_and = np.count_nonzero(AND > 0)
    C0_or = np.count_nonzero(OR > 0)
    if C0_or != 0:
        IoU_dec = C0_and / C0_or

    return IoU_dec


def SKL_process(skeleton, endpoints):
    # get all skeletons
    SKLS = []
    for i in range(len(endpoints)):
        end_i = endpoints[i]
        skeleton_i = np.copy(skeleton)
        PF = []
        SKLS = get_allSK(skeleton_i, [end_i], PF, SKLS)

    # delete repeted
    rpt = list(np.ones((1, len(SKLS))))
    for i in range(len(SKLS)-1):
        BW_SKL_i = build_skel(skeleton.shape[0], skeleton.shape[1], SKLS[i])
        for j in range(i+1, len(SKLS)):
            if int(rpt[0][j]) == 1:
                BW_SKL_j = build_skel(skeleton.shape[0], skeleton.shape[1], SKLS[j])
                BW_SKL = abs(BW_SKL_i - BW_SKL_j)
                if BW_SKL.max() == 0:
                    rpt[0][j] = 0

    # save not repeted
    SKLS_NR = []
    for i in range(len(SKLS)):
        if int(rpt[0][i]) == 1:
            SKLS_NR.append(SKLS[i])

    smth = []
    lnw = []
    cwi = []
    for i in range(len(SKLS_NR)):
        SKL_i = SKLS_NR[i]
        lnw.append(len(SKL_i))
        s1 = 0
        for j in range(len(SKL_i) - 2):
            y1, x1 = SKL_i[j]
            y2, x2 = SKL_i[j + 1]
            y3, x3 = SKL_i[j + 2]
            degrees1 = math.atan2(x1 - x2, y1 - y2)
            degrees2 = math.atan2(x1 - x3, y1 - y3)
            dg = abs(degrees1 - degrees2)
            s1 = s1 + dg
        smth.append(s1)
        cwi.append(s1 * (1 / len(SKL_i)))

    i_worm = cwi.index(min(cwi))
    SKL_true = SKLS_NR[i_worm]
    BW_SKL_j = build_skel(skeleton.shape[0], skeleton.shape[1], SKL_true)
    # plt.imshow(BW_SKL_j)
    # plt.show()

    return SKL_true


def chk_endPoints_edge(worm, endpoints):
    chkL = 0
    dt_worm = ndimage.distance_transform_edt(worm)

    dt_endPoint = []
    for i in range(len(endpoints)):
        XY = endpoints[i]
        dt_endPoint.append(dt_worm[XY[0], XY[1]])

    for i in range(len(endpoints)-1):
        dt_i = dt_endPoint[i]
        for j in range(i+1, len(endpoints)):
            dt_j = dt_endPoint[j]
            if abs(dt_i-dt_j) > 4:
                chkL = 1
    # plt.imshow(BW_SKL_j)
    # plt.show()

    return chkL


def chk_endPoints_edge1(worm, endpoints):
    chkL = 0
    dt_worm = ndimage.distance_transform_edt(worm)
    mask_edge = (edge_img(worm) > 0) * 1

    edge_dtimg = dt_worm * mask_edge
    if edge_dtimg.max() > 4:
        chkL = 1
    # plt.imshow(BW_SKL_j)
    # plt.show()
    return chkL


def skeleton_check_edge(img):
    masK_edge = edge_img(img)
    # skeleton = morphology.medial_axis(img) * 255
    skeleton_fix = cv2.bitwise_and(img.astype('uint8'), cv2.bitwise_not(masK_edge))
    return skeleton_fix > 0


def process_mask(masks):
    f_worm = 0
    lenghts = []
    worms_an = []
    for i in range(masks.shape[0]):
        worm = masks[i, :, :]
        worm = true_worm(worm)  # remove small parts
        skeleton = skeletonize(worm) * 255  # skeletonize worm
        skeleton = skeleton_check_edge(skeleton) * 255
        # skeleton2 = skeleton_medial_axis(worm) * 255  # skeletonize worm
        branchs, endpoints = branchs_endpoints(skeleton)
        # chk_edge = edge_worm(worm)
        chk_edge = chk_endPoints_edge1(worm, endpoints)

        if chk_edge == 0:  # is not on the edge
            if len(endpoints) == 2 and len(branchs) == 0:
                f_worm = 1
            else:
                chk_w = check_skl(skeleton, branchs)
                if chk_w == 1:
                    f_worm = 1
                    # skl = SKL_process(skeleton, endpoints)  # best skeleton ----------------------------------------

        if f_worm == 1:
            lenghts.append(np.count_nonzero(skeleton))
        else:
            lenghts.append(0)
        worms_an.append(f_worm)
        f_worm = 0

    if len(masks.shape) > 2:
        masK_worms = np.zeros((masks.shape[1], masks.shape[2]), dtype="uint8")
        worms_true = np.zeros((masks.shape[1], masks.shape[2]), dtype="uint8")
        worms_bads = np.zeros((masks.shape[1], masks.shape[2]), dtype="uint8")
    else:
        masK_worms = np.zeros((masks.shape[0], masks.shape[1]), dtype="uint8")
        worms_true = np.zeros((masks.shape[0], masks.shape[1]), dtype="uint8")
        worms_bads = np.zeros((masks.shape[0], masks.shape[1]), dtype="uint8")
    max_lenght = max(lenghts)

    index_good = []
    index_bad = []
    for i in range(len(worms_an)):
        worm = masks[i, :, :].astype('uint8')
        if worms_an[i] == 1:
            pLenght = lenghts[i]/max_lenght
            if pLenght > 0.75:
                masK_worms = cv2.bitwise_or(masK_worms, worm)
                worms_true = update_worms(worms_true, worm)
                index_good.append(i)
            else:
                # worms_bads = cv2.bitwise_or(worms_bads, worm)
                worms_bads = update_worms(worms_bads, worm)
                index_bad.append(i)
        else:
            # worms_bads = cv2.bitwise_or(worms_bads, worm)
            worms_bads = update_worms(worms_bads, worm)
            index_bad.append(i)

    # rgbArray = np.zeros((masks.shape[1], masks.shape[2], 3), 'uint8')
    # rgbArray[..., 2] = masK_worms * 255
    # rgbArray[..., 0] = worms_bads * 255
    #
    # plt.imshow(rgbArray)
    # plt.show()

    results = {
        # *****************
        # [masks]
        # *****************
        'worms_true': worms_true,
        'worms_bads': worms_bads,
        # *****************
        # [labels]
        # *****************
        'index_good': index_good,
        'index_bad': index_bad
    }
    return results


def change_masks(prediction_mask, index_good):
    all_elements = []
    for i in range(prediction_mask.shape[0]):
        all_elements.append(str(i))

    all_elements = all_elements + index_good
    index_bad = [i for i in all_elements if all_elements.count(i) == 1]

    worms_true = np.zeros((prediction_mask.shape[1], prediction_mask.shape[2]), np.int8)
    for i in range(len(index_good)):
        try:
            index = int(index_good[i])
            worms_true = update_worms(worms_true, prediction_mask[index, :, :])
        except:
            rt = -1

    worms_bad = np.zeros((prediction_mask.shape[1], prediction_mask.shape[2]), np.int8)
    for i in range(len(index_bad)):
        try:
            index = int(index_bad[i])
            worms_bad = update_worms(worms_bad, prediction_mask[index, :, :])
        except:
            rt = -1

    results = {
        # *****************
        # [masks]
        # *****************
        'worms_true': worms_true,
        'worms_bads': worms_bad,
        # *****************
        # [labels]
        # *****************
        'index_good': index_good,
        'index_bad': index_bad
    }

    return results

