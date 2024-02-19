# Import the necessary libraries and modules
import os
import argparse
from unet_models import UnetSegmentation,StyleUnetGenerator,NLayerDiscriminator,DeepSea
from cellpose_model import Cellpose_CPnet
from utils import set_requires_grad,mixed_list,noise_list,image_noise,initialize_weights
from data import BasicDataset,get_image_mask_pairs
import itertools
import torch.nn as nn
from tqdm import tqdm
from evaluate import evaluate_segmentation
from loss import CombinedLoss,VGGLoss
import torch.utils.data as data
import torch.nn.functional as F
import transforms as transforms
import torch
import numpy as np
import random
import logging
from diffaug import DiffAugment
from torch.optim.lr_scheduler import ReduceLROnPlateau
# Set a constant seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# Function to reset logging configuration
def reset_logging():
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

# The main training function
def train(args,image_size = [512,768],image_means = [0.5],image_stds= [0.5],train_ratio = 0.8,save_checkpoint=True):

    # Reset logging configuration
    reset_logging()

    # Set up the logging
    logging.basicConfig(filename=os.path.join(args.output_dir, 'train.log'), filemode='w',
                        format='%(asctime)s - %(message)s', level=logging.INFO)
    logging.info('>>>> image size=(%d,%d) , learning rate=%f , batch size=%d' % (
        image_size[0], image_size[1], args.lr, args.batch_size))

    # Determine the device (GPU or CPU) to use
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define the transforms for the training and validation data
    train_transforms = transforms.Compose([
                               transforms.ToPILImage(),
                               transforms.RandomApply([transforms.RandomOrder([
                                   transforms.RandomApply([transforms.ColorJitter(brightness=0.33, contrast=0.33, saturation=0.33, hue=0)],p=0.5),
                                   transforms.RandomApply([transforms.GaussianBlur((5, 5), sigma=(0.1, 1.0))],p=0.5),
                                   transforms.RandomApply([transforms.RandomHorizontalFlip(0.5)],p=0.5),
                                   transforms.RandomApply([transforms.RandomVerticalFlip(0.5)],p=0.5),
                                   transforms.RandomApply([transforms.AddGaussianNoise(0., 0.01)], p=0.5),
                                   transforms.RandomApply([transforms.CLAHE()], p=0.5),
                                   transforms.RandomApply([transforms.RandomAdjustSharpness(sharpness_factor=2)], p=0.5),
                                   transforms.RandomApply([transforms.RandomCrop()], p=0.5),
                                ])],p=args.p_vanilla),
                               transforms.Resize(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize(mean = image_means,std = image_stds)
                           ])

    dev_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=image_means,std=image_stds)
    ])
    # Read samples
    sample_pairs=get_image_mask_pairs(args.train_set_dir)
    assert len(sample_pairs)>0, f'No samples found in {args.train_set_dir}'

    # Split samples
    train_sample_pairs=sample_pairs[:int(train_ratio*len(sample_pairs))]
    valid_sample_pairs=sample_pairs[int(train_ratio * len(sample_pairs)):]

    # Define the datasets for training and validation
    train_data = BasicDataset(train_sample_pairs,transforms=train_transforms,vanilla_aug=False if args.p_vanilla==0 else True,gen_nc=args.gen_nc)
    valid_data = BasicDataset(valid_sample_pairs,transforms=dev_transforms,gen_nc=args.gen_nc)

    # Define the dataloaders
    train_iterator = data.DataLoader(train_data,shuffle = True,batch_size = args.batch_size,num_workers=8,pin_memory=True)
    valid_iterator = data.DataLoader(valid_data,batch_size = args.batch_size,num_workers=8 ,pin_memory=True)

    # Define the models
    Gen = StyleUnetGenerator(style_latent_dim = 128,output_nc=args.gen_nc)
    if args.seg_model=='UNET':
        Seg = UnetSegmentation(n_channels=args.gen_nc, n_classes=2)
    elif args.seg_model=='CellPose':
        Seg = Cellpose_CPnet(n_channels=args.gen_nc,n_classes=2)
    elif args.seg_model=='DeepSea':
        Seg = DeepSea(n_channels=args.gen_nc, n_classes=2)
    else:
        # If none of the above models are matched, raise an error
        raise ValueError(f"Model '{args.seg_model}' not found.")

    D1 = NLayerDiscriminator(input_nc=args.gen_nc)
    D2 = NLayerDiscriminator(input_nc=1)

    initialize_weights(Gen)
    initialize_weights(Seg)
    initialize_weights(D1)
    initialize_weights(D2)

    # Define the optimizers
    optimizer_G = torch.optim.Adam(itertools.chain(Gen.parameters(), Seg.parameters()), lr=args.lr)
    optimizer_D1 = torch.optim.Adam(D1.parameters(), lr=args.lr)
    optimizer_D2 = torch.optim.Adam(D2.parameters(), lr=args.lr)

    grad_scaler = torch.cuda.amp.GradScaler(enabled=True)

    # Define the loss functions
    d_criterion = nn.MSELoss()
    Gen_criterion_1=nn.L1Loss()
    Gen_criterion_2=VGGLoss()
    Seg_criterion = CombinedLoss()

    Gen = Gen.to(device)
    Seg = Seg.to(device)
    D1 = D1.to(device)
    D2 = D2.to(device)

    d_criterion=d_criterion.to(device)
    Gen_criterion_1=Gen_criterion_1.to(device)
    Gen_criterion_2 = Gen_criterion_2.to(device)
    Seg_criterion = Seg_criterion.to(device)

    # Training loop
    nstop=0
    avg_precision_best=0
    logging.info('>>>> Start training')
    print('INFO: Start training ...')
    for epoch in range(args.max_epoch):
        Gen.train()
        Seg.train()
        D1.train()
        D2.train()

        for step,batch in enumerate(tqdm(train_iterator)):
            real_img = batch['image']
            real_mask = batch['mask']

            valid = torch.full((real_mask.shape[0], 1, 62, 94), 1.0, dtype=real_mask.dtype, device=device)
            fake = torch.full((real_mask.shape[0], 1, 62, 94), 0.0, dtype=real_mask.dtype, device=device)

            real_img = real_img.to(device=device, dtype=torch.float32)
            real_mask = real_mask.to(device=device, dtype=torch.float32)

            set_requires_grad(D1, True)
            set_requires_grad(D2, True)
            set_requires_grad(Gen, True)
            set_requires_grad(Seg, True)

            optimizer_G.zero_grad(set_to_none=True)  # Clear gradients
            optimizer_D1.zero_grad(set_to_none=True)
            optimizer_D2.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=True):
                if random.random() < 0.9:
                    style = mixed_list(real_img.shape[0], 5, Gen.latent_dim, device=device)
                else:
                    style = noise_list(real_img.shape[0], 5, Gen.latent_dim, device=device)

                im_noise = image_noise(real_mask.shape[0], image_size, device=device)

                fake_img = Gen(real_mask,style, im_noise)
                rec_mask= Seg(fake_img)
                fake_mask = Seg(real_img)
                fake_mask_p = F.softmax(fake_mask, dim=1).float()
                fake_mask_p = torch.unsqueeze(fake_mask_p.argmax(dim=1), dim=1)
                fake_mask_p=fake_mask_p.to(dtype=torch.float32)

                if random.random() < 0.9:
                    style = mixed_list(real_mask.shape[0], 4, Gen.latent_dim, device=device)
                else:
                    style = noise_list(real_mask.shape[0], 4, Gen.latent_dim, device=device)

                im_noise = image_noise(real_mask.shape[0], image_size, device=device)

                rec_img = Gen(fake_mask_p,style, im_noise)

                set_requires_grad(D1, False)
                set_requires_grad(D2, False)

                d_img_loss = d_criterion(D1(DiffAugment(fake_img,p=args.p_diff)), valid)
                d_mask_loss = d_criterion(D2(fake_mask_p), valid)
                rec_mask_loss=100 * Seg_criterion(rec_mask, torch.squeeze(real_mask.to(dtype=torch.long), dim=1))
                id_mask_loss = 50 * Seg_criterion(fake_mask, torch.squeeze(real_mask.to(dtype=torch.long), dim=1))
                rec_img_loss=100 * Gen_criterion_1(rec_img, real_img)+100 * Gen_criterion_2(rec_img, real_img)
                id_img_loss = 50 * Gen_criterion_1(fake_img, real_img)+50 * Gen_criterion_2(fake_img, real_img)
                g_loss=d_mask_loss+d_img_loss+rec_mask_loss+rec_img_loss+id_mask_loss+id_img_loss

                grad_scaler.scale(g_loss).backward()  # Scale the loss, and then backward pass
                grad_scaler.step(optimizer_G)  # Update optimizer with scaled gradients
                grad_scaler.update()  # Update the scale for next iteration

                set_requires_grad(D1, True)
                set_requires_grad(D2, True)

                real_img_loss = d_criterion(D1(DiffAugment(real_img,p=args.p_diff)), valid)
                fake_img_loss = d_criterion(D1(DiffAugment(fake_img.detach(),p=args.p_diff)), fake)
                d_img_loss = (real_img_loss + fake_img_loss) / 2

                grad_scaler.scale(d_img_loss).backward()
                grad_scaler.step(optimizer_D1)
                grad_scaler.update()

                real_mask_loss = d_criterion(D2(real_mask), valid)
                fake_mask_loss = d_criterion(D2(fake_mask_p.detach()), fake)
                d_mask_loss = (real_mask_loss + fake_mask_loss) / 2

                grad_scaler.scale(d_mask_loss).backward()
                grad_scaler.step(optimizer_D2)
                grad_scaler.update()


        print(
            "[Epoch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, args.max_epoch, d_mask_loss.item()+d_img_loss.item(), g_loss.item())
        )

        # Evaluate the model and save the best checkpoint
        val_score,avg_precision,avg_val_loss = evaluate_segmentation(Seg, valid_iterator, device,Seg_criterion,len(valid_data),is_avg_prec=True,prec_thresholds=[0.5],output_dir=None)
        if avg_precision is not None:
            logging.info('>>>> Epoch:%d  , Dice score=%f , avg precision=%f' % (epoch,val_score, avg_precision[0]))
        else:
            logging.info('>>>> Epoch:%d  , Dice score=%f' % (epoch,val_score))

        if avg_precision is not None and avg_precision>avg_precision_best:
            avg_precision_best=avg_precision
            if save_checkpoint:
                torch.save(Gen.state_dict(), os.path.join(args.output_dir, 'Gen.pth'))
                torch.save(Seg.state_dict(), os.path.join(args.output_dir, 'Seg.pth'))
                torch.save(D1.state_dict(), os.path.join(args.output_dir, 'D1.pth'))
                torch.save(D2.state_dict(), os.path.join(args.output_dir, 'D2.pth'))
                logging.info('>>>> Save the model checkpoints to %s'%(os.path.join(args.output_dir)))
            nstop=0
        elif avg_precision is not None and avg_precision<=avg_precision_best:
            nstop+=1
        if nstop==args.patience:#Early Stopping
            print('INFO: Early Stopping met ...')
            print('INFO: Finish training process')
            break


# Define the main function
if __name__ == "__main__":
    # Define the argument parser
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_set_dir",required=True,type=str,help="path for the train dataset")
    ap.add_argument("--lr", default=1e-4,type=float, help="learning rate")
    ap.add_argument("--max_epoch", default=2000, type=int, help="maximum epoch to train model")
    ap.add_argument("--batch_size", default=2, type=int, help="train batch size")
    ap.add_argument("--output_dir", required=True, type=str, help="path for saving the train log and best checkpoint")
    ap.add_argument("--p_vanilla", default=0.2,type=float, help="probability value of vanilla augmentation")
    ap.add_argument("--p_diff", default=0.2,type=float, help="probability value of diff augmentation, a value between 0 and 1")
    ap.add_argument("--seg_model", required=True, type=str, help="segmentation model type (DeepSea or CellPose or UNET)")
    ap.add_argument("--patience",default=500, type=int, help="Number of patience epochs for early stopping")
    ap.add_argument("--gen_nc", default=1, type=int, help="1 for 2D or 3 for 3D, the number of generator output channels")

    # Parse the command-line arguments
    args = ap.parse_args()

    # Check if the test set directory exists
    assert os.path.isdir(args.train_set_dir), 'No such file or directory: ' + args.train_set_dir
    # If output directory doesn't exist, create it
    os.makedirs(args.output_dir, exist_ok=True)

    train(args)