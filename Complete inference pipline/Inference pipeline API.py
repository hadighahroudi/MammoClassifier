from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from pipeline_utils import create_convnextv2_base, remove_pect_muscle, prep_convnextv2_for_gradcam,\
    predict, grad_cam_plusplus, get_org_cc_mlo_maps
import cv2
from UNet3Plus import ResNet101UNet3Plus
import torch
from torchvision.transforms import v2
import os

SEGMENTOR_PATH = r"D:\Study\Proposal\Breast cancer(v2)\MammoClassifier\checkpoints\last_resnet101_unet3-inbreast_mias-breast_roi-adam-no_cls_guide-no_mixup-elastic_flip-output_resized-bs8_e100.pt"
CLASSIFIER_PATH = r"D:\Study\Proposal\Breast cancer(v2)\MammoClassifier\checkpoints\convnextv2_base-AdamW-up_sample-pos_smooth-mixup-cmmd_vindr-VOILUT_Flipped_pect_imgs-bs8x8-s0_e36_seed0.pt" 

# Define the request body model
class PredictionRequest(BaseModel):
    cc_path: str
    mlo_path: str

def prediction_pipeline(request: PredictionRequest):
    print("[INFO] Removing pect muscle...")
    mlo_roi, mlo_bbox, mlo_org_shape = remove_pect_muscle(request.mlo_path, segmentor_transform, resize_to_org, segmentor_model, device)
    print("[INFO] Predicting breast cancer...")
    logits, probs, cc_mlo_img_org, cc_mlo_img, cc_bbox, cc_org_shape = predict(request.cc_path, mlo_roi, classifier_model, device, classifier_transform)
    print("[INFO] Generating saliency maps...")
    image_with_map, heatmap = grad_cam_plusplus(logits, classifier_model, cc_mlo_img_org, cc_mlo_img, device)
    print("[INFO] Saving outputs...")
    cc_map, mlo_map = get_org_cc_mlo_maps(heatmap, cc_bbox, mlo_bbox, cc_org_shape, mlo_org_shape)

    cc_map_path = os.path.splitext(request.cc_path)[0] + "_map.png" 
    mlo_map_path = os.path.splitext(request.mlo_path)[0] + "_map.png"
    cv2.imwrite(cc_map_path, cv2.cvtColor(cc_map, cv2.COLOR_RGB2BGR))
    cv2.imwrite(mlo_map_path, cv2.cvtColor(mlo_map, cv2.COLOR_RGB2BGR))

    return cc_map_path, mlo_map_path, logits.tolist(), probs.tolist()

# Initialize the FastAPI app
app = FastAPI()

origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

segmentor_model = ResNet101UNet3Plus(num_classes = 2,
                    resnet_weights = None,
                    class_guided = False,
                    is_batchnorm = True,
                    output_size = (512, 512)).to(device) 
segmentor_model.load_state_dict(torch.load(SEGMENTOR_PATH, map_location=device))
classifier_model = create_convnextv2_base(device, CLASSIFIER_PATH)
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

# Endpoint to handle requests
@app.post("/predict")
def calculate(request: PredictionRequest):
    try:
        print("[INFO] Processing the request...")
        cc_map_path, mlo_map_path, logits, probs = prediction_pipeline(request)

        return {"cc_map_path": cc_map_path,
                "mlo_map_path": mlo_map_path,
                "logits": logits,
                "probs": probs}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str('test error'))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
