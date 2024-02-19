# Import necessary libraries and packages
import os
import torch.utils.data as data
import transforms as transforms
import numpy as np
import argparse
import random
from loss import CombinedLoss
from unet_models import DeepSea,UnetSegmentation
from cellpose_model import Cellpose_CPnet
from data import BasicDataset,get_image_mask_pairs  # Import the Dataset handling module
import torch
from evaluate import evaluate_segmentation  # Import the evaluation function

# Set up seeds for reproducible results
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


# Define the testing function
def test(args, image_size=[512, 768], image_means=[0.5], image_stds=[0.5], batch_size=1):
    # Determine if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define the transformation to be applied on the images
    test_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=image_means, std=image_stds)
    ])

    # Read samples
    sample_pairs = get_image_mask_pairs(args.test_set_dir)
    assert len(sample_pairs)>0, f'No samples found in {args.test_set_dir}'

    # Load the test dataset and apply the transformations
    test_data = BasicDataset(sample_pairs,transforms=test_transforms)

    # Create a dataloader for the test dataset
    test_iterator = data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Create an instance of the Segmentation model and load the trained model
    if args.seg_model == 'UNET':
        Seg = UnetSegmentation(n_channels=1, n_classes=2).to(device)    # Seg = nn.DataParallel(DeepSea_old(n_channels=1, n_classes=2, bilinear=True)).to(device)
    elif args.seg_model=='CellPose':
       Seg = Cellpose_CPnet(n_channels=1, n_classes=2).to(device)
    elif args.seg_model=='DeepSea':
        Seg = DeepSea(n_channels=1, n_classes=2).to(device)
    else:
        # If none of the above models are matched, raise an error
        raise ValueError(f"Model '{args.seg_model}' not found.")


    Seg.load_state_dict(torch.load(args.seg_ckpt_dir))
    # Seg.load_state_dict(torch.load(args.seg_ckpt_dir))

    # Define the loss functions
    Seg_criterion = CombinedLoss()

    # Evaluate the model and calculate the dice score and average precision
    scores = evaluate_segmentation(Seg, test_iterator, device,Seg_criterion, len(test_data),
                                                           is_avg_prec=True, prec_thresholds=[0.5, 0.6, 0.7, 0.8, 0.9],
                                                           output_dir=args.output_dir)


    return scores

# Define the main function
if __name__ == "__main__":
    # Define the argument parser
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_set_dir", required=True, type=str, help="path for the test dataset")
    ap.add_argument("--seg_ckpt_dir", required=True, type=str, help="path for the checkpoint of segmentation model to test")
    ap.add_argument("--output_dir", required=True, type=str, help="path for saving the test outputs")
    ap.add_argument("--seg_model", required=True, type=str, help="segmentation model type (DeepSea or CellPose or UNET)")

    # Parse the command-line arguments
    args = ap.parse_args()

    # Check if the test set directory exists
    assert os.path.isdir(args.test_set_dir), 'No such file or directory: ' + args.test_set_dir

    # Run the test function
    scores=test(args)

    # Print scores
    print('Average Dice score:', scores['dice_score'])
    print('Average loss:', scores['avg_val_loss'])
    print('Average precision at ordered thresholds:', scores['avg_precision'])
    print('Average recall at ordered thresholds:', scores['avg_recall'])
    print('Average fscore at ordered thresholds:', scores['avg_fscore'])
