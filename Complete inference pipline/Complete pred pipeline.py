from pipeline_utils import create_convnextv2_base, remove_pect_muscle, prep_convnextv2_for_gradcam,\
    predict, grad_cam_plusplus, get_org_cc_mlo_maps, remove_artifacts, process_dcm_image, heatmap_transparent, array_info
import cv2
from UNet3Plus import ResNet101UNet3Plus
import torch
from torchvision.transforms import v2
import argparse
import numpy as np
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cc_path", type=str)
    parser.add_argument("--mlo_path", type=str)
    parser.add_argument("--segmentor_path", type=str)
    parser.add_argument("--classifier_path", type=str)
    parser.add_argument("--output_cc_path", type=str)
    parser.add_argument("--output_mlo_path", type=str)
    args = parser.parse_args()

    segmentor_path = args.segmentor_path
    classifier_path = args.classifier_path
    cc_path = args.cc_path
    mlo_path = args.mlo_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    segmentor_model = ResNet101UNet3Plus(num_classes = 2,
                       resnet_weights = None,
                       class_guided = False,
                       is_batchnorm = True,
                       output_size = (512, 512)).to(device) 
    segmentor_model.load_state_dict(torch.load(segmentor_path, map_location=device))


    classifier_model = create_convnextv2_base(device, classifier_path)
    classifier_model = prep_convnextv2_for_gradcam(classifier_model)


    segmentor_transform = v2.Compose([
        v2.ToImage(),
        v2.Resize(size = (512, 512), antialias = True),
        v2.ToDtype(torch.float32, scale = True)
        ])

    resize_to_org = v2.Resize((1024, 512), antialias = True)

    classifier_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize([0.20275, 0.20275, 0.20275],
                    [0.19875, 0.19875, 0.19875])])

    print("[INFO] Removing pect muscle...")
    mlo_roi, mlo_bbox, mlo_org_shape = remove_pect_muscle(args.mlo_path, segmentor_transform, resize_to_org, segmentor_model, device)

    print("[INFO] Predicting breast cancer...")
    logits, probs, cc_mlo_img_org, cc_mlo_img, cc_bbox, cc_org_shape = predict(args.cc_path, mlo_roi, classifier_model, device, classifier_transform)
    # assert mlo_lat == cc_lat, "The mammograms' laterality does not match."

    print("[INFO] Generating saliency maps...")
    image_with_org_map, heatmap = grad_cam_plusplus(logits, classifier_model, cc_mlo_img_org, cc_mlo_img, device)
    heatmap_a = np.dstack((heatmap, np.full(heatmap.shape[:-1], 255, dtype = np.uint8)))
    heatmap_a = heatmap_transparent(heatmap_a)
    heatmap_a = remove_artifacts(heatmap_a, cc_mlo_img)

    print("[INFO] Saving outputs...")
    cc_map, mlo_map = get_org_cc_mlo_maps(heatmap_a, cc_bbox, mlo_bbox, cc_org_shape, mlo_org_shape)
    # if mlo_lat == "L":
    #     cc_map, mlo_map = cv2.flip(cc_map, 1), cv2.flip(mlo_map, 1)

    cc_map_path = os.path.splitext(args.cc_path)[0] + "_map.png" 
    mlo_map_path = os.path.splitext(args.mlo_path)[0] + "_map.png"
    cv2.imwrite(cc_map_path, cv2.cvtColor(cc_map, cv2.COLOR_RGBA2BGRA))
    cv2.imwrite(mlo_map_path, cv2.cvtColor(mlo_map, cv2.COLOR_RGBA2BGRA))

    print(f"Logits: {logits}, Probs: {probs}")
    print(f"Ouput images saved to {args.output_cc_path}, {args.output_mlo_path}")

    # Showing the original image overlayed with heatmaps
    cc_img_org = (process_dcm_image(args.cc_path)[0] * 255).astype(np.uint8)
    mlo_img_org = (process_dcm_image(args.mlo_path)[0] * 255).astype(np.uint8)

    cc_img_org = cv2.cvtColor(cc_img_org, cv2.COLOR_GRAY2BGRA)
    cc_map = cv2.cvtColor(cc_map, cv2.COLOR_RGBA2BGRA)
    cc_img_map_org = (cc_img_org * 0.5 + cc_map * 0.5).astype(np.uint8)

    mlo_img_org = cv2.cvtColor(mlo_img_org, cv2.COLOR_GRAY2BGRA)
    mlo_map = cv2.cvtColor(mlo_map, cv2.COLOR_RGBA2BGRA)
    mlo_img_map_org = (mlo_img_org * 0.5 + mlo_map * 0.5).astype(np.uint8)
    array_info(mlo_img_map_org, "cc_img_map_org")
    cv2.imshow("cc_img_map_org", cv2.resize(mlo_img_map_org, (256, 512)))
    cv2.waitKey(0)

    
