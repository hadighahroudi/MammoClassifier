import cv2
import pydicom
from pydicom.pixels import apply_modality_lut, apply_voi_lut
import numpy as np
import pywt
from types import MethodType
import torch
from torch import nn
import timm
from tqdm.auto import tqdm
import os
from typing import Optional, List
from scipy.ndimage import zoom

def crop_roi(image, threshold: Optional[int] = None, bounds = (50, 50, 50, 50)): # ToDo: make it accepts channel first images resulted by ToImage transform
    if image.dtype != np.uint8:
        image = (min_max_norm(image)[0] * 255).astype(np.uint8)
        
    image = image[bounds[0]:-bounds[1], bounds[2]:-bounds[3]]
    
    if threshold:
        thresh_image = (image > threshold).astype(np.uint8)
    else:
        thresh_value, thresh_image = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)
        
    numLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh_image, 8, cv2.CV_32S)
    idx = stats[1:, 4].argmax() + 1 # stats[0]: backgournd stats, stats[:, 4]: Area of connected commponents
                                    # We add 1 to the argmax since the first component with highest area is the whole image
    labels[labels != idx] = 0
    labels[labels == idx] = 1
    labels = labels[:, :, np.newaxis].astype(np.uint8) # labels dtype is int32
    
    image = labels * image # to remove image annotations surrounding breast
    
    x1, y1, w, h = stats[idx][:4]
    x2 = x1 + w
    y2 = y1 + h
    
    return image[y1: y2, x1: x2], (x1+bounds[2], y1+bounds[0], x2+bounds[2], y2+bounds[0])


def create_convnextv2_base(device, checkpoint_path=None, compile = False):
    model = timm.create_model('convnextv2_base')

    model.head.fc = nn.Sequential(
        nn.Linear(in_features = 1024, out_features=128, bias = True),
        nn.Linear(in_features = 128, out_features = 2, bias = True)
    )

    if compile:
        model = torch.compile(model)
    
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        print(f"Loading {checkpoint_path}")
        print(model.load_state_dict(checkpoint))

    torch.cuda.empty_cache()
    model.to(device)

    return model
    

class DWT2_CLAHE(object):
    def __init__(self, wavelet='db9'):
        self.wavelet = wavelet
    
    def __call__(self, image):
        if isinstance(image, np.ndarray) == False:
            image = np.asarray(image)

        # Remove channels dims
        n_ch = image.shape[-1]
        if n_ch == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) 
        elif n_ch == 1:
            image = image.squeeze()

        coeffs = pywt.wavedec2(image, self.wavelet, level=2)

        cA2 = coeffs[0]
        cH2 = coeffs[1][0]
        cV2 = coeffs[1][1]
        cD2 = coeffs[1][2]
        cH1 = coeffs[2][0]
        cV1 = coeffs[2][1]
        cD1 = coeffs[2][2]

        cH2 = np.zeros(cH2.shape)
        cV2 = np.zeros(cV2.shape)
        cD2 = np.zeros(cD2.shape)
        cH1 = np.zeros(cH1.shape)
        cV1 = np.zeros(cV1.shape)
        cD1 = np.zeros(cD1.shape)

        clahe = cv2.createCLAHE(clipLimit = 2)
        cA2_norm, cA2_max, cA2_min = min_max_norm(cA2)
        cA2_clahe = clahe.apply((cA2_norm * 255).astype("uint8"))
        cA2_clahe_norm, _, _ = min_max_norm(cA2_clahe)
        cA2_clahe_denorm = min_max_denorm(cA2_clahe_norm, cA2_max, cA2_min)

        new_coeffs = [cA2_clahe_denorm, (cH2, cV2, cD2), (cH1, cV1, cD1)]

        rec_image = pywt.waverec2(new_coeffs, self.wavelet)

        rec_image_norm, _, _ = min_max_norm(rec_image)

        rec_image_norm = (rec_image_norm * 255).astype(np.uint8)

        # if rec_image_norm.shape != image.shape:
        #     rec_image_norm = cv2.resize(rec_image_norm, image.shape[::-1])
        
        if n_ch == 3:
            rec_image_norm = np.stack((rec_image_norm, rec_image_norm, rec_image_norm), -1) # Add the channels dim back

        return rec_image_norm
    
def process_dcm_image(path):
    dcm = pydicom.dcmread(path)
    img = apply_modality_lut(dcm.pixel_array, dcm)
    img = apply_voi_lut(img, dcm)
    img = min_max_norm(img)[0]
    if dcm.PhotometricInterpretation == "MONOCHROME1":
        img = 1 - img

    return img, img.shape

def get_laterality(img):
    mid = img.shape[1] // 2
    if img[:, :mid].sum() < img[:, mid:].sum():
        return "R"
    else:
        return "L"

def remove_pect_muscle(mlo_path, preprocess_transform, postprocess_transform, segmentor_model, device):
    # Remove pect muscle
    mlo_img, mlo_org_shape = process_dcm_image(mlo_path)
    mlo_img, mlo_bbox = crop_roi(mlo_img[:, :, np.newaxis], threshold=20)
    # mlo_img = DWT2_CLAHE()(mlo_img)
    mlo_img = cv2.resize(mlo_img, (512, 1024))

    lat = get_laterality(mlo_img)
    if lat == "L":
       mlo_img = cv2.flip(mlo_img, 1)

    image = preprocess_transform(mlo_img)
    image = min_max_norm(image)[0]
    image = torch.stack((image, image, image), dim = 1) # (1, 512, 512) -> (1, 3, 512, 512)

    with torch.inference_mode():
        outputs_dict = segmentor_model(image.to(device))
        prob = torch.sigmoid(outputs_dict["final_pred"][:, 1])
        pred = torch.round(prob).type(torch.uint8)

    pred = postprocess_transform(pred).cpu().numpy()
    mlo_roi = (1-pred).squeeze() * mlo_img.squeeze()

    return mlo_roi, mlo_bbox, mlo_org_shape, lat


def prep_convnextv2_for_gradcam(classifier_model):
    # Based on: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/convnext.py
    
#     def forward_features(self, x):
#         x = self.stem(x)
#         x = self.stages(x)
#         x = self.norm_pre(x)
#         return x

#     def forward_head(self, x, pre_logits: bool = False):
#         return self.head(x, pre_logits=True) if pre_logits else self.head(x)

    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        x = self.forward_features(x)
        
        # register the hook
        h = x.register_hook(self.activations_hook)
        
        x = self.forward_head(x)
        return x

    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self, x):
        return self.forward_features(x)
    

    classifier_model.activations_hook = MethodType(activations_hook, classifier_model)
    classifier_model.get_activations_gradient = MethodType(get_activations_gradient, classifier_model)
    classifier_model.get_activations = MethodType(get_activations, classifier_model)
    classifier_model.forward = MethodType(forward, classifier_model)

    return classifier_model

def predict(cc_path, mlo_roi, classifier_model, device, transform):
    # Predict the probs
    cc_img, cc_org_shape = process_dcm_image(cc_path) 
    cc_img, cc_bbox = crop_roi(cc_img[:, :, np.newaxis], threshold=20)
    # cc_img = DWT2_CLAHE()(cc_img)
    cc_img = cv2.resize(cc_img, (512, 1024))

    lat = get_laterality(cc_img)
    if lat == "L":
       cc_img = cv2.flip(cc_img, 1)

    cc_mlo_img_org = np.concatenate([cc_img, mlo_roi], axis = 1)

    cc_mlo_img = transform(cc_mlo_img_org)
        
    classifier_model.eval()
    # with torch.inference_mode():
    logits = classifier_model(cc_mlo_img.unsqueeze(0).to(device))
    probs = torch.softmax(logits, dim = 1)
        
    return logits, probs, cc_mlo_img_org, cc_mlo_img, cc_bbox, cc_org_shape, lat


def grad_cam(logits, classifier_model, cc_mlo_img_org, cc_mlo_img, device):
    # get the gradient of the output with respect to the parameters of the model
    logits[:, logits.argmax()].backward()
    # pull the gradients out of the model
    gradients = classifier_model.get_activations_gradient()
    # pool the gradients across the channels
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    # get the activations of the last convolutional layer
    activations = classifier_model.get_activations(cc_mlo_img.unsqueeze(0).to(device)).detach()
    # weight the channels by corresponding gradients
    for i in range(len(pooled_gradients)):
        activations[:, i, :, :] *= pooled_gradients[i]
    # average the channels of the activations
    heatmap = torch.mean(activations, dim=1).squeeze()
    # relu on top of the heatmap
    # expression (2) in https://arxiv.org/pdf/1610.02391.pdf
    heatmap = torch.relu(heatmap)
    # normalize the heatmap
    heatmap /= torch.max(heatmap)
    # interpolate the heat-map and project it onto the original image
    resized_heatmap = cv2.resize(heatmap.cpu().numpy(), (cc_mlo_img.shape[-1], cc_mlo_img.shape[-2]))
    resized_heatmap = np.uint8(255 * resized_heatmap)
    resized_heatmap = cv2.applyColorMap(resized_heatmap, cv2.COLORMAP_JET)

    rgb_img = np.stack([cc_mlo_img_org, cc_mlo_img_org, cc_mlo_img_org], axis=2)
    superimposed_img = resized_heatmap * 0.4 + rgb_img # Values may go beyond 255
    superimposed_img = np.uint8(min_max_norm(superimposed_img)[0] * 255) 

    return superimposed_img

def scale_cam_image(cam, target_size=None):
    result = []
    for img in cam:
        img = img - np.min(img)
        img = img / (1e-7 + np.max(img))
        if target_size is not None:
            if len(img.shape) > 2:
                img = zoom(np.float32(img), [
                           (t_s / i_s) for i_s, t_s in zip(img.shape, target_size[::-1])])
            else:
                img = cv2.resize(np.float32(img), target_size)

        result.append(img)
    result = np.float32(result)

    return result


# Adopted from: https://github.com/jacobgil/pytorch-grad-cam/blob/master/pytorch_grad_cam/base_cam.py#L141
# and https://github.com/jacobgil/pytorch-grad-cam/blob/master/pytorch_grad_cam/grad_cam_plusplus.py#L7
def grad_cam_plusplus(logits, classifier_model, cc_mlo_img_org, cc_mlo_img, device):
    # get the gradient of the output with respect to the parameters of the model
    logits[:, logits.argmax()].backward()
    # pull the gradients out of the model
    grads = classifier_model.get_activations_gradient()
    grads_power_2 = grads ** 2
    grads_power_3 = grads ** 3

    # get the activations of the last convolutional layer
    activations = classifier_model.get_activations(cc_mlo_img.unsqueeze(0).to(device)).detach()
    # Equation 19 in https://arxiv.org/abs/1710.11063
    sum_activations = torch.sum(activations, dim=(2, 3))
    eps = 0.000001
    aij = grads_power_2 / (2 * grads_power_2 + sum_activations[:, :, None, None] * grads_power_3 + eps)

    # Now bring back the ReLU from eq.7 in the paper,
    # And zero out aijs where the activations are 0
    aij = torch.where(grads != 0, aij, 0)

    weights = torch.relu(grads) * aij
    weights = torch.sum(weights, dim=(2, 3))

    # 2D conv
    if len(activations.shape) == 4:
        weighted_activations = weights[:, :, None, None] * activations
    # 3D conv
    elif len(activations.shape) == 5:
        weighted_activations = weights[:, :, None, None, None] * activations

    heatmap = weighted_activations.sum(axis=1)
    heatmap = torch.relu(heatmap)
    scaled_heatmap = scale_cam_image(heatmap.cpu().numpy(), (cc_mlo_img.shape[-1], cc_mlo_img.shape[-2])).squeeze()

    # interpolate the heat-map and project it onto the original image
    scaled_heatmap = np.uint8(255 * scaled_heatmap)
    scaled_heatmap = cv2.applyColorMap(scaled_heatmap, cv2.COLORMAP_JET)

    rgb_img = np.stack([cc_mlo_img_org, cc_mlo_img_org, cc_mlo_img_org], axis=2)
    superimposed_img = scaled_heatmap * 0.4 + rgb_img # Values may go beyond 255
    superimposed_img = np.uint8(min_max_norm(superimposed_img)[0] * 255) 

    return superimposed_img, scaled_heatmap
    

def heatmap_transparent(heatmap, start_color = [128, 0, 0, 255], end_color = [128, 0, 0, 255]):
    """Makes the heatmap background transparent"""
    # Define the replacement range
    replacement_start = np.array(start_color[:-1]+ [0])
    replacement_end = np.array(end_color[:-1] + [0])
    
    # Define color ranges for substitution
    start_color = np.array(start_color)  # Starting RGBA range
    end_color = np.array(end_color)    # Ending RGBA range
    
    # Create a mask for pixels within the range
    mask = np.all((heatmap >= start_color) & (heatmap <= end_color), axis=-1)
    
    # Replace the colors for matching pixels
    heatmap[mask] = replacement_start + (heatmap[mask] - start_color)

    return heatmap

def remove_artifacts(heatmap, cc_mlo_img, threshold = 0):
    """Removes any hot area outside the breast tissue"""
    mask = ((min_max_norm(cc_mlo_img)[0] * 255)[0] > threshold).type(torch.uint8).numpy()
    mask = cv2.dilate(mask, (5, 5), iterations = 5)[..., np.newaxis]
    return heatmap * mask

def get_org_cc_mlo_maps(dualview_heatmap, cc_bbox, mlo_bbox, cc_org_shape, mlo_org_shape):
    """
    This function pads the saliency maps to match the original image size
    Note: The dualview_heatmap is supposed to have transparent background.
    """
    w = dualview_heatmap.shape[1]
    # 1. Seperate CC and MLO heatmaps
    cc_map, mlo_map = dualview_heatmap[:, :w//2], dualview_heatmap[:, w//2:]

    h, w = cc_org_shape
    x1, y1, x2, y2 = cc_bbox
    # 2. Resize each heatmap to the size after crop roi
    cc_map = cv2.resize(cc_map, (x2-x1, y2-y1))
    # 3. Pad the map to match the dicom image
    cc_map = cv2.copyMakeBorder(cc_map, y1, h-y2, x1, w-x2, cv2.BORDER_CONSTANT, value = (128, 0, 0, 0))
    # 4. Smoothen the conjuction of border and map
    
    h, w = mlo_org_shape
    x1, y1, x2, y2 = mlo_bbox
    # 2. Resize each heatmap to the size after crop roi
    mlo_map = cv2.resize(mlo_map, (x2-x1, y2-y1))
    # 3. Pad the map to match the dicom image
    mlo_map = cv2.copyMakeBorder(mlo_map, y1, h-y2, x1, w-x2, cv2.BORDER_CONSTANT, value = (128, 0, 0, 0))

    return cc_map, mlo_map



def scantree(path, exts: List[str] = None):
    for entity in os.scandir(path):
        if entity.is_dir(follow_symlinks = False):
            yield from scantree(entity.path, exts)
        else:
            if exts != None and os.path.splitext(entity.path)[-1] in exts:
                yield entity.path
            elif exts == None:
                yield entity.path


def min_max_norm(a):
    a_norm = (a - a.min()) / (a.max() - a.min())
    return a_norm, a.max(), a.min()

def min_max_denorm(a_norm, a_max, a_min):
    return (a_norm * (a_max - a_min)) + a_min

def array_info(arr, name: str = ""):
    print(f"{name}\nType: {type(arr)}\nShape: {arr.shape}\ndtype: {arr.dtype}\nmin/max: {arr.min()}/{arr.max()}")