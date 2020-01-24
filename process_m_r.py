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


classes = ['10_p', '11_q', '12_r', '13_s', '13_s_rotated', '14_t', '15_e', '16_k', '17_u', '18_v', '19_w', '1_a', '20_f',
           '20_f_rotated', '21_g', '22_g', '23_h', '24_h', '25_f', '25_f_rotated', '26_f', '27_x', '27_x_rotated', '28_i',
           '28_i_rotated', '29_i', '2_a', '30_i', '30_i_rotated', '31_j', '32_j', '33_e', '34_k', '35_k', '36_i', 
           '36_i_rotated', '37_i', '38_l', '39_l', '3_b', '40_l', '41_y', '42_z', '42_z_rotated', '43_m', '44_m', 
           '44_m_rotated', '45_i', '45_i_rotated', '46_n', '46_n_rotated', '47_n', '47_n_rotated', '48_n', '48_n_rotated', 
           '49_aa', '4_b', '50_n', '51_n', '51_n_rotated', '52_ab', '53_ac', '54_ad', '55_l', '56_ae', '57_af', '58_i', 
           '59_o', '5_c', '60_o', '60_o_rotated', '6_ag', '6_ag_rotated', '7_c', '8_d', '9_d']

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

learner = load_learner(path='.',file='pcb_msng_and_rotd.pkl',device='cpu')
model = learner.model.cpu()

img_read = get_prediction_image(sys.argv[1])
with torch_no_grad():
    output = model(img_read)
    
bboxes, preds, scores = get_predictions(output,0, detect_thresh=0.015)

pred_labels = [classes[pred] for pred in preds]

missing = set(gt_labels) - set(pred_labels)
rotated = set(pred_labels) - set(gt_labels)

if(len(missing.union(rotated)) == 0):
    print("No Errors")
    
for mc in missing:
    print(f"Component {mc.split('_')[0]} Missing")
for rot in rotated:
    print(f"Component {rot.split('_')[0]} Rotated")