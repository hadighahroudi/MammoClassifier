from pipeline_utils import process_dcm_image, crop_roi, get_org_cc_mlo_maps
import numpy as np
import cv2



# mlo_path = r"D:\Study\Proposal\Breast cancer(v2)\MammoClassifier\Complete inference pipline\test_images\20587994_024ee3569b2605dc_MG_R_CC_ANON.dcm"
# mlo_img_org, mlo_org_shape = process_dcm_image(mlo_path)
# mlo_img, mlo_bbox = crop_roi(mlo_img_org[:, :, np.newaxis] , threshold=20)

# cc_path = r"D:\Study\Proposal\Breast cancer(v2)\MammoClassifier\Complete inference pipline\test_images\20588046_024ee3569b2605dc_MG_R_ML_ANON.dcm"
# cc_img_org, cc_org_shape = process_dcm_image(cc_path)
# cc_img, cc_bbox = crop_roi(cc_img_org[:, :, np.newaxis] , threshold=20)

# heatmap = cv2.imread(r"D:\Study\Proposal\Breast cancer(v2)\MammoClassifier\Complete inference pipline\map.png")
# cc_map, mlo_map = get_org_cc_mlo_maps(heatmap, cc_bbox, mlo_bbox, cc_org_shape, mlo_org_shape)

# cv2.imwrite("temp_cc.png", np.uint8(mlo_img_org*255))
# cv2.imwrite("temp_mlo.png", np.uint8(cc_img_org*255))

# cv2.imwrite("temp_cc_map.png", cc_map)
# cv2.imwrite("temp_mlo_map.png", mlo_map)

# cc_map = cv2.imread("map_cc.png")
# k_size = (20, 20)
# x1 = 2522
# cc_map[:, x1-k_size[0]:x1+k_size[1]] = cv2.blur(cc_map[:, x1-k_size[0]:x1+k_size[1]], k_size)
# cv2.imwrite("new_cc_map.png", cc_map)

path = r"D:\Study\Proposal\Breast cancer(v2)\MammoClassifier\MammoClassifier.Web\wwwroot\studies\maps\I0000002_map.png"
img = cv2.imread(path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
# Create a mask for pixels that match the target color
mask = (
    (img[:, :, 0] == 0) &  # Blue channel
    (img[:, :, 1] == 0) &  # Green channel
    (img[:, :, 2] < 255)  # Red channel
)
img[mask, 3] = 0 # Set the alpha channel to 0 for the matching pixels

cv2.imwrite(path, img)