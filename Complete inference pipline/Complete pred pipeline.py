from pipeline_utils import create_convnextv2_base, remove_pect_muscle, prep_convnextv2_for_gradcam,\
    predict, grad_cam_plusplus, get_org_cc_mlo_maps, remove_artifacts
import cv2
from UNet3Plus import ResNet101UNet3Plus
import torch
from torchvision.transforms import v2
import argparse


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

    mlo_roi, mlo_bbox, mlo_org_shape = remove_pect_muscle(mlo_path, segmentor_transform, resize_to_org, segmentor_model, device)
    logits, probs, cc_mlo_img_org, cc_mlo_img, cc_bbox, cc_org_shape = predict(cc_path, mlo_roi, classifier_model, device, classifier_transform)
    image_with_map, heatmap = grad_cam_plusplus(logits, classifier_model, cc_mlo_img_org, cc_mlo_img, device)
    heatmap, cc_mlo_img = remove_artifacts(heatmap, cc_mlo_img)
    cc_map, mlo_map = get_org_cc_mlo_maps(heatmap, cc_bbox, mlo_bbox, cc_org_shape, mlo_org_shape)

    cv2.imwrite(args.output_cc_path, cv2.cvtColor(cc_map, cv2.COLOR_RGB2BGR))
    cv2.imwrite(args.output_mlo_path, cv2.cvtColor(mlo_map, cv2.COLOR_RGB2BGR))
    cv2.imwrite("image_with_map.png", cv2.cvtColor(image_with_map, cv2.COLOR_RGB2BGR))
    # cv2.imwrite(args.output_path, cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))

    print(f"Logits: {logits}, Probs: {probs}")
    print(f"Ouput images saved to {args.output_cc_path}, {args.output_mlo_path}")
