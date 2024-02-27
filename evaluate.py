import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from skimage.morphology import remove_small_objects
import copy
import cv2
import os
from loss import multiclass_dice_coeff
from scipy.optimize import linear_sum_assignment
from utils import visualize_segmentation
from scipy import ndimage as ndi


def evaluate_segmentation(net, valid_iterator, device,criterion,n_valid_examples,is_avg_prec=False,prec_thresholds=[0.5,0.7,0.9],output_dir=None):
    if output_dir is not None:
        os.makedirs(os.path.join(output_dir,'input_images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'segmentation_predictions'), exist_ok=True)
    net.eval()
    num_val_batches = len(valid_iterator)
    dice_score = 0
    mask_list, pred_list= [], []
    # iterate over the validation set
    with tqdm(total=n_valid_examples, desc='Segmentation Val round', unit='img') as pbar:
        total_val_loss=0
        for batch_idx,batch in enumerate(valid_iterator):
            images, true_masks = batch['image'], batch['mask']
            images_device = images.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=torch.long)
            true_masks = torch.squeeze(true_masks, dim=1)
            true_masks_copy = copy.deepcopy(true_masks)
            true_masks_one_hot = F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).float()

            with torch.no_grad():
                # predict the mask
                mask_preds= net(images_device)
                loss = criterion(mask_preds, true_masks)
                total_val_loss += loss.item()
                # convert to one-hot format
                mask_pred_copy = copy.deepcopy(mask_preds.argmax(dim=1))
                mask_preds = F.one_hot(mask_preds.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_preds[:, 1:, ...], true_masks_one_hot[:, 1:, ...],
                                                    reduce_batch_first=False)
                if is_avg_prec:
                    true_masks_copy=true_masks_copy.cpu().numpy()
                    mask_pred_copy = mask_pred_copy.cpu().numpy()
                    for i in range(true_masks_copy.shape[0]):
                        mask,_=ndi.label(remove_small_objects(true_masks_copy[i,:,:]>0,min_size=15,connectivity=1))
                        mask_list.append(mask)
                        pred, _ = ndi.label(remove_small_objects(mask_pred_copy[i, :, :]>0,min_size=15,connectivity=1))
                        if output_dir:
                            img=images[i].cpu().numpy()[0, :, :]
                            img=(img-np.min(img))/(np.max(img)-np.min(img))*255
                            overlay_img = visualize_segmentation(pred, inp_img=img, overlay_img=True)
                            cv2.imwrite(os.path.join(output_dir,'input_images','images_{:04d}.png'.format(batch_idx*true_masks_copy.shape[0]+i)),visualize_segmentation(mask, inp_img=img, overlay_img=True))
                            cv2.imwrite(os.path.join(output_dir, 'segmentation_predictions', 'images_{:04d}.png'.format(batch_idx*true_masks_copy.shape[0]+ i)),overlay_img)
                        pred_list.append(pred)

            pbar.update(images.shape[0])

    avg_val_loss = total_val_loss / len(valid_iterator)
    if is_avg_prec:
        precision_list,recall_list,fscore_list=average_precision_recall_fscore(mask_list, pred_list, threshold=prec_thresholds)

        scores={
            'dice_score':dice_score.cpu().numpy() / num_val_batches,
            'avg_precision':np.mean(precision_list, axis=0),
            'avg_recall': np.mean(recall_list, axis=0),
            'avg_fscore': np.mean(fscore_list, axis=0),
            'avg_val_loss':avg_val_loss

        }
        return scores

    scores = {
        'dice_score': dice_score.cpu().numpy() / num_val_batches,
        'avg_precision': None,
        'avg_recall': None,
        'avg_fscore': None,
        'avg_val_loss': avg_val_loss
    }
    return scores


def _label_overlap(x, y):
    """ fast function to get pixel overlaps between masks in x and y

    Parameters
    ------------

    x: ND-array, int
        where 0=NO masks; 1,2... are mask labels
    y: ND-array, int
        where 0=NO masks; 1,2... are mask labels

    Returns
    ------------

    overlap: ND-array, int
        matrix of pixel overlaps of size [x.max()+1, y.max()+1]

    """
    x = x.ravel()
    y = y.ravel()
    overlap = np.zeros((1 + x.max(), 1 + y.max()), dtype=np.uint)
    for i in range(len(x)):
        overlap[x[i], y[i]] += 1
    return overlap

def _intersection_over_union(masks_true, masks_pred):
    """ intersection over union of all mask pairs

    Parameters
    ------------

    masks_true: ND-array, int
        ground truth masks, where 0=NO masks; 1,2... are mask labels
    masks_pred: ND-array, int
        predicted masks, where 0=NO masks; 1,2... are mask labels

    Returns
    ------------

    iou: ND-array, float
        matrix of IOU pairs of size [x.max()+1, y.max()+1]

    """
    overlap = _label_overlap(masks_true, masks_pred)
    n_pixels_pred = np.sum(overlap, axis=0, keepdims=True)
    n_pixels_true = np.sum(overlap, axis=1, keepdims=True)
    iou = overlap / (n_pixels_pred + n_pixels_true - overlap+1e-6)
    iou[np.isnan(iou)] = 0.0
    return iou


def _true_positive(iou, th):
    """ true positive at threshold th

    Parameters
    ------------

    iou: float, ND-array
        array of IOU pairs
    th: float
        threshold on IOU for positive label

    Returns
    ------------

    tp: float
        number of true positives at threshold

    """
    n_min = min(iou.shape[0], iou.shape[1])
    costs = -(iou >= th).astype(float) - iou / (2 * n_min+1e-6)
    true_ind, pred_ind = linear_sum_assignment(costs)
    match_ok = iou[true_ind, pred_ind] >= th
    tp = match_ok.sum()
    return tp



def average_precision_recall_fscore(masks_true, masks_pred, threshold=[0.5, 0.75, 0.9]):

    not_list = False
    if not isinstance(masks_true, list):
        masks_true = [masks_true]
        masks_pred = [masks_pred]
        not_list = True
    if not isinstance(threshold, list) and not isinstance(threshold, np.ndarray):
        threshold = [threshold]
    recall = np.zeros((len(masks_true), len(threshold)), np.float32)
    precision = np.zeros((len(masks_true), len(threshold)), np.float32)
    fscore = np.zeros((len(masks_true), len(threshold)), np.float32)
    tp = np.zeros((len(masks_true), len(threshold)), np.float32)
    fp = np.zeros((len(masks_true), len(threshold)), np.float32)
    fn = np.zeros((len(masks_true), len(threshold)), np.float32)
    n_true = np.array(list(map(np.max, masks_true)))
    n_pred = np.array(list(map(np.max, masks_pred)))
    with tqdm(total=len(masks_true), desc='Metrics measurement', unit='img') as pbar:
        for n in range(len(masks_true)):
            if n_pred[n] > 0:
                iou = _intersection_over_union(masks_true[n], masks_pred[n])[1:, 1:]
                for k, th in enumerate(threshold):
                    tp[n, k] = _true_positive(iou, th)
            fp[n] = n_pred[n] - tp[n]
            fn[n] = n_true[n] - tp[n]
            recall[n] = tp[n] / (tp[n] + fn[n] + 1e-6)
            precision[n] = tp[n] / (tp[n] + fp[n] + 1e-6)
            fscore[n] = 2 * (precision[n] * recall[n]) / (precision[n] + recall[n] + 1e-6)

            pbar.update(1)
    if not_list:
        precision,recall,fscore, tp, fp, fn = precision[0],recall[0],fscore[0], tp[0], fp[0], fn[0]

    return precision,recall,fscore