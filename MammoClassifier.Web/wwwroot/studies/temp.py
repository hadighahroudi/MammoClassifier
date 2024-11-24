import cv2
import pydicom
import numpy as np

map = cv2.imread(r"D:\Study\Proposal\Breast cancer(v2)\MammoClassifier\MammoClassifier.Web\wwwroot\studies\maps\I0000000_map.png")
img = pydicom.dcmread(r"D:\Study\Proposal\Breast cancer(v2)\MammoClassifier\MammoClassifier.Web\wwwroot\studies\images\8684388\I0000000.dcm").pixel_array
img = (img - img.min()) / (img.max() - img.min()) 
img = (1 - img) * 255
img = img.astype(np.uint8)
img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

cv2.imwrite("temp.png", img * 0.5 + map * 0.5)