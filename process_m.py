from fastai.vision import load_learner, Image, pil2tensor
from torch import no_grad as torch_no_grad
from cv2 import imread, split, merge
from detection_network import get_predictions
import sys
import numpy as np
from os.path import isfile
from ast import literal_eval

print()
if len(sys.argv) == 1:
        print("Provide an image to run the model")
        print()
        print(f"Usage: python {sys.argv[0]} <pcb_image_path>\n")
        sys.exit(1)

if not isfile(sys.argv[1]):
    print(f"{sys.argv[1]} does not exist!!\n")
    sys.exit(1)

classes = ['10_p', '4_b', '36_i', '37_i', '39_l', '41_y', '42_z', '43_m', '44_m', '45_i', '51_n',
           '30_i', '55_l', '57_af', '59_o', '5_c', '60_o', '6_ag', '7_c', '8_d', '33_e', '38_l', '2_a',
           '25_f', '18_v', '19_w', '1_a', '14_t', '13_s', '22_g', '23_h', '24_h', '21_g', '17_u', '28_i',
           '26_f', '12_r', '27_x', '29_i', '40_l', '32_j', '15_e', '46_n', '34_k', '35_k', '20_f', '50_n',
           '56_ae', '3_b', '58_i', '52_ab', '16_k', '53_ac', '47_n', '48_n', '31_j', '11_q', '9_d', '54_ad', '49_aa']

with open("./DLAssignment/number_to_type.json") as read_file:
    json_str = read_file.read()
label_dict = literal_eval(json_str)
gt_labels = [k+'_'+v for k,v in label_dict.items()]


def get_prediction_image(path,sz=256):
    bgr_img = imread(path)
    b,g,r = split(bgr_img)
    rgb_img = merge([r,g,b])
    rgb_img = rgb_img/255.0
    img = Image(px=pil2tensor(rgb_img, np.float32))
    img = img.resize((3,sz,sz))
    return img.px.reshape(1,3,sz,sz)

learner = load_learner(path='.',file='export.pkl',device='cpu')
model = learner.model.cpu()

img_read = get_prediction_image(sys.argv[1])
with torch_no_grad():
    output = model(img_read)
    
bboxes, preds, scores = get_predictions(output,0, detect_thresh=0.015)

pred_labels = [classes[pred] for pred in preds]

missing = set(gt_labels) - set(pred_labels)

if(len(missing.union(rotated)) == 0):
    print("No Errors")
for mc in missing:
    print(f"Component {mc.split('_')[0]} Missing")