import os
import numpy as np
import cv2


im_dir = './img_celeba'
out_dir = './celeba_cropped'
bbox_fpath = './celeba_crop_bbox.txt'
out_im_size = 128

im_list = np.loadtxt(bbox_fpath, dtype='str')
total_num = len(im_list)
split_dict = {'0': 'train',
              '1': 'val',
              '2': 'test'}

for i, row in enumerate(im_list):
    if i%1000 == 0:
        print(f'{i}/{total_num}')

    fname = row[0]
    split = row[1]
    x0, y0, w, h = row[2:].astype(int)
    im = cv2.imread(os.path.join(im_dir, fname))
    im_pad = cv2.copyMakeBorder(im, h, h, w, w, cv2.BORDER_REPLICATE)  # allow cropping outside by replicating borders
    im_crop = im_pad[y0+h:y0+h*2, x0+w:x0+w*2]
    im_crop = cv2.resize(im_crop, (out_im_size,out_im_size))

    out_folder = os.path.join(out_dir, split_dict[split])
    os.makedirs(out_folder, exist_ok=True)
    cv2.imwrite(os.path.join(out_folder, fname), im_crop)
