import random
import cv2
import numpy as np
from torchvision.models import inception_v3
from scipy.linalg import sqrtm
import torch
import torch.nn.init as init
import torch.nn as nn
import os
from PIL import Image

DATA_LOADER_SEED = 9
random.seed(DATA_LOADER_SEED)
class_colors = [(0,0,0)]+[(random.randint(50, 255), random.randint(
    50, 255), random.randint(50, 255)) for _ in range(1000)]

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            # He initialization for layers with ReLU activation
            init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            # Initializing BatchNorm layers
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            # Xavier initialization for fully connected layers
            init.xavier_normal_(m.weight)
            init.constant_(m.bias, 0)

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def get_colored_segmentation_image(seg_arr, n_classes, colors=class_colors):
    output_height = seg_arr.shape[0]
    output_width = seg_arr.shape[1]

    seg_img = np.zeros((output_height, output_width, 3))

    for c in range(n_classes):
        seg_arr_c = seg_arr[:, :] == c
        seg_img[:, :, 0] += ((seg_arr_c)*(colors[c][0])).astype('uint8')
        seg_img[:, :, 1] += ((seg_arr_c)*(colors[c][1])).astype('uint8')
        seg_img[:, :, 2] += ((seg_arr_c)*(colors[c][2])).astype('uint8')

    return seg_img

def overlay_seg_image(inp_img, seg_img):
    orininal_h = inp_img.shape[0]
    orininal_w = inp_img.shape[1]
    if len(inp_img.shape)==2:
        inp_img=cv2.cvtColor(inp_img, cv2.COLOR_GRAY2BGR)
    seg_img = cv2.resize(seg_img, (orininal_w, orininal_h))

    fused_img = (inp_img/2 + seg_img/2).astype('uint8')
    return fused_img

def get_legends(class_names, colors=class_colors):

    n_classes = len(class_names)
    legend = np.zeros(((len(class_names) * 25) + 25, 125, 3),
                      dtype="uint8") + 255

    class_names_colors = enumerate(zip(class_names[:n_classes],
                                       colors[:n_classes]))

    for (i, (class_name, color)) in class_names_colors:
        color = [int(c) for c in color]
        cv2.putText(legend, class_name, (5, (i * 25) + 17),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)
        cv2.rectangle(legend, (100, (i * 25)), (125, (i * 25) + 25),
                      tuple(color), -1)

    return legend

def concat_lenends(seg_img, legend_img):

    new_h = np.maximum(seg_img.shape[0], legend_img.shape[0])
    new_w = seg_img.shape[1] + legend_img.shape[1]

    out_img = np.zeros((new_h, new_w, 3)).astype('uint8') + legend_img[0, 0, 0]

    out_img[:legend_img.shape[0], :  legend_img.shape[1]] = np.copy(legend_img)
    out_img[:seg_img.shape[0], legend_img.shape[1]:] = np.copy(seg_img)

    return out_img

def visualize_segmentation(seg_arr, inp_img=None, n_classes=None,
                           colors=class_colors, class_names=None,
                           overlay_img=False, show_legends=False,
                           prediction_width=None, prediction_height=None):

    if n_classes is None:
        n_classes = np.max(seg_arr)+1

    seg_img = get_colored_segmentation_image(seg_arr, n_classes, colors=colors)

    if inp_img is not None:
        orininal_h = inp_img.shape[0]
        orininal_w = inp_img.shape[1]
        seg_img = cv2.resize(seg_img, (orininal_w, orininal_h))

    if (prediction_height is not None) and (prediction_width is not None):
        seg_img = cv2.resize(seg_img, (prediction_width, prediction_height))
        if inp_img is not None:
            inp_img = cv2.resize(inp_img,
                                 (prediction_width, prediction_height))

    if overlay_img:
        assert inp_img is not None
        seg_img = overlay_seg_image(inp_img, seg_img)

    if show_legends:
        assert class_names is not None
        legend_img = get_legends(class_names, colors=colors)
        seg_img = concat_lenends(seg_img, legend_img)

    return seg_img

def calculate_fid(images1, images2,device):
    # Convert images to PyTorch tensors if they are not
    if not torch.is_tensor(images1):
        images1 = torch.tensor(images1)
    if not torch.is_tensor(images2):
        images2 = torch.tensor(images2)

    # Ensure the images tensors are float and on the right device
    images1 = images1.float().to(device)
    images2 = images2.float().to(device)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).eval().to(device)  # Move model to device

    # Extract features for both image sets
    with torch.no_grad():
        pred1 = inception_model(images1)
        pred2 = inception_model(images2)

    # Compute mean and covariance for both sets
    mu1, sigma1 = pred1.mean(0), torch_cov(pred1)
    mu2, sigma2 = pred2.mean(0), torch_cov(pred2)

    # Compute sum of squared difference between the means
    ssdiff = torch.sum((mu1 - mu2)**2.0)

    # Compute sqrt of product between cov
    covmean = sqrtm((sigma1.cpu().numpy() + sigma2.cpu().numpy()) / 2)

    # Check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    # Compute the final FID score
    fid = ssdiff + torch.trace(sigma1 + sigma2 - 2*torch.tensor(covmean).to(sigma1.device))

    return fid.item()

def torch_cov(m, rowvar=False):
    '''Estimate a covariance matrix given data.

    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element `C_{ij}` is the covariance of
    `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.

    Args:
        m: A 1-D or 2-D array containing multiple variables and observations.
            Each row of `m` represents a variable, and each column a single
            observation of all those variables.
        rowvar: If `rowvar` is True, then each row represents a
            variable, with observations in the columns. Otherwise, the
            relationship is transposed: each column represents a
            variable, while the rows contain observations.

    Returns:
        The covariance matrix of the variables.
    '''
    if m.dim() > 2:
        raise ValueError('m has more than 2 dimensions')
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()
    # m = m.type(torch.double)  # uncomment this line if desired
    fact = 1.0 / (m.size(1) - 1)
    m -= torch.mean(m, dim=1, keepdim=True)
    mt = m.t()  # if complex: mt = m.t().conj()
    return fact * m.matmul(mt).squeeze()

def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

def noise(n, latent_dim, device):
    return torch.randn(n, latent_dim).cuda(device)

def noise_list(n, layers, latent_dim, device):
    return [(noise(n, latent_dim, device), layers)]

def mixed_list(n, layers, latent_dim, device):
    tt = int(torch.rand(()).numpy() * layers)
    return noise_list(n, tt, latent_dim, device) + noise_list(n, layers - tt, latent_dim, device)

def image_noise(n, im_size, device):
    return torch.FloatTensor(n, im_size[0], im_size[1], 1).uniform_(0., 1.).cuda(device)


# Function to apply weights within a circle
def apply_weights_in_circle(weights, center, radius, weight_val,map_height, map_width):
    for y in range(max(0, center[1] - radius), min(map_height, center[1] + radius + 1)):
        for x in range(max(0, center[0] - radius), min(map_width, center[0] + radius + 1)):
            if (x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius ** 2:
                weights[y, x] = weight_val
    return weights

def prepare_weights(map_height, map_width,num_regions):
    # Flat or colony-like with a 50% chance
    if random.random() > 0.5: #Flat
        weights = np.ones((map_height, map_width))
    else: #colony-like
        # Initialize the weights array
        weights = np.zeros((map_height, map_width))
        # Apply random weighted regions
        for _ in range(num_regions):
            # Random center for the circle
            center = (random.randint(0, map_width - 1), random.randint(0, map_height - 1))

            # Random radius
            radius = random.randint(10, 400)  # Adjust min and max radius as needed

            # Random weight increase
            weight_val = 1  # Adjust as needed
            # Apply weights within the circle
            weights = apply_weights_in_circle(weights, center, radius, weight_val,map_height, map_width)

    # Normalize the weights
    flat_weights = weights.flatten()
    flat_weights /= flat_weights.sum()
    return flat_weights


def apply_mask_augmentations(mask):
    mask = Image.fromarray(mask)

    # Flip horizontally with a 50% chance
    if random.random() > 0.5:
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

    # Flip vertically with a 50% chance
    if random.random() > 0.5:
        mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

    # Rotate randomly between 0 and 360 degrees
    rotation_angle = random.randint(0, 360)
    mask = mask.rotate(rotation_angle, fillcolor=0)  # Assuming mask background is 0 for transparency

    mask = np.array(mask)
    return mask

def save_generated_synthetic_mask(dest_dir,new_mask,mask_id):
        if np.sum(new_mask):
            new_mask[new_mask > 0] = 255
            cv2.imwrite(os.path.join(dest_dir, 'img_{:04d}.png'.format(mask_id)), new_mask)

def apply_cell_body_mask_augmentation(cropped_cell_body):
    # Resize with a 50% chance
    if random.random() > 0.5:#resize
        scale_x = 0.75 + random.randint(0, 50) / 100
        scale_y = 0.75 + random.randint(0, 50) / 100
        cropped_mask = cv2.resize(cropped_cell_body.astype(np.uint8), (0, 0), fx=scale_x, fy=scale_y)
        cropped_mask[cropped_mask < 0.5] = 0
        cropped_mask[cropped_mask >= 0.5] = 1
    return cropped_cell_body