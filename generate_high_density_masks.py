import argparse
import os
import cv2
import shutil
import numpy as np
from scipy import ndimage as ndi
from tqdm import tqdm
from utils import prepare_weights,apply_mask_augmentations,\
    save_generated_synthetic_mask,apply_cell_body_mask_augmentation


def generate_synthetic_masks(source_dir,dest_dir):
    generated_mask_id = 0
    new_mask = 0
    num_repetition_per_orig_dataset=2
    num_repetition_per_orig_img=2
    num_orig_masks_to_merge = 5
    num_merged_masks=-1
    for _ in range(num_repetition_per_orig_dataset):
        #Loop over original mask images
        for i, img_name in enumerate(tqdm(sorted(os.listdir(source_dir)))):
            for _ in range(num_repetition_per_orig_img):
                num_merged_masks+=1
                mask = cv2.imread(os.path.join(source_dir, img_name))
                mask = (255 * ((mask - mask.min()) / mask.ptp())).astype(np.uint8)
                mask=apply_mask_augmentations(mask)
                markers, num_labels = ndi.label(mask)
                if num_merged_masks % (num_orig_masks_to_merge*num_repetition_per_orig_img) == 0:
                    save_generated_synthetic_mask(dest_dir,new_mask,generated_mask_id)
                    generated_mask_id+=1
                    new_mask = np.zeros(mask.shape)
                    map_height, map_width = markers.shape[0], markers.shape[1]
                    flat_weights = prepare_weights(map_height, map_width, num_regions=3)

                for label in np.unique(markers):
                    if label == 0:
                        continue
                    cell_body_mask = markers.copy()
                    cell_body_mask[cell_body_mask != label] = 0
                    cell_body_mask[cell_body_mask > 0] = 1
                    y_nonzero, x_nonzero = np.nonzero(cell_body_mask[:, :, 0])
                    w = np.max(x_nonzero) - np.min(x_nonzero)
                    h = np.max(y_nonzero) - np.min(y_nonzero)

                    #is a valid cell body?
                    if w < 15 or h < 15 or np.min(y_nonzero) < 2 or np.min(x_nonzero) < 2 or np.max(y_nonzero) > (
                            cell_body_mask.shape[0] - 3) or np.max(x_nonzero) > (cell_body_mask.shape[1] - 3):
                        markers[markers == label] = 0
                        continue

                    cropped_cell_body = cell_body_mask.copy()[np.min(y_nonzero):np.max(y_nonzero), np.min(x_nonzero):np.max(x_nonzero)]

                    h, w, c = cropped_cell_body.shape
                    for _ in range(100):  #Try to find an empy place for the cell bbody on the new mask not overlaping others
                        cropped_cell_body = apply_cell_body_mask_augmentation(cropped_cell_body)
                        # Select a potential center point based on the weights
                        selected_index = np.random.choice(np.arange(map_height * map_width), p=flat_weights)
                        y0, x0 = divmod(selected_index, map_width)
                        x0 = min(max(x0, 2), map_width - w - 2)
                        y0 = min(max(y0, 2), map_height - h - 2)

                        cell_body_mask[cell_body_mask > 0] = 0
                        overlap_detected = False
                        for c in range(w):
                            for r in range(h):
                                cell_body_mask[y0 + r, x0 + c] = cropped_cell_body[r, c]
                                if new_mask[y0 + r, x0 + c, 0] != 0:  # Is there any overlap with other previously generated cell bodies
                                    overlap_detected = True
                                    break
                        if overlap_detected==False:
                            new_mask += cell_body_mask # Add cell body to the generated synthetic mask
                            break

# Define the main function
if __name__ == "__main__":
    # Argument parsing
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_mask_set_dir", required=True, type=str, help="path for the real mask image set as the input")
    ap.add_argument("-synthetic_mask_set_dir", required=True, type=str, help="path for saving the generated mask images")

    args = ap.parse_args()

    # Check if test dataset directory exists
    assert os.path.isdir(args.real_mask_set_dir), 'No such file or directory: ' + args.real_mask_set_dir

    if os.path.exists(args.synthetic_mask_set_dir):
        # Remove the existing directory and all its contents
        shutil.rmtree(args.synthetic_mask_set_dir)

    # Create the new directory
    os.makedirs(args.synthetic_mask_set_dir)

    generate_synthetic_masks(args.real_mask_set_dir, args.synthetic_mask_set_dir)