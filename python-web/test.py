#from PIL import Image
#from lib._rgb2ycbcr import *

#from scipy.misc import imread
#from scipy.misc import imresize

import os, json, argparse
import numpy as np
import tensorflow as tf
from lib._data import *
from lib._predict import *

from scipy import misc
from skimage import color

if __name__ == '__main__':
	with tf.Session() as session:
		network = predict.load_model(session)

		for set_name in ['Set5', 'Set14', 'B100', 'Urban100']:
			for scaling_factor in [2, 3, 4]:
				dataset = data.TestSet(set_name, scaling_factors=[scaling_factor])
				predictions, psnr = predict.predict(dataset.images, session, network, targets=dataset.targets,
													border=scaling_factor)

				print('Dataset "%s", scaling factor = %d. Mean PSNR = %.2f.' % (set_name, scaling_factor, np.mean(psnr)))

'''
	f_size = 96
	s_epoch = 27

	DATA_PATH = "_uploads/extracts/"
	result_path_dir = "_uploads/result/"
	file_list = os.listdir(DATA_PATH)
	file_list.sort()

	image= Image.open("_uploads/extracts/extract_449.jpg")
	ycbcr = image.convert('YCbCr')
	width = image.size[1]
	height = image.size[0]
	B_ycbcr = np.ndarray((image.size[1], image.size[0], 3), 'uint8', ycbcr.tobytes())
	B_ycbcr2 = np.ndarray((image.size[1], image.size[0], 3), 'uint8', ycbcr.tobytes())
	B_ycbcr2 = cropping(B_ycbcr2, height, width)
	B_ycbcr2_raw = B_ycbcr2[:,:,0]/255.0
	#print (B_ycbcr.shape)
	wpercent = (width/float(image.size[0]))
	hsize = int((float(image.size[1])*float(wpercent)))
	img = image.resize((width*2,hsize*2), Image.ANTIALIAS)
	img.save("_uploads/extracts/extract_449X2_result.jpg")
	print (B_ycbcr2_raw)

	print("\n\n\n")
	image2 = imread("_uploads/extracts/extract_449.jpg")
	ycbcr2 = rgb2ycbcr(image2).astype('uint8')[:,:,0]/255.0
	width = image2.shape[1]
	height = image2.shape[0]
	im_gt_ycbcr = rgb2ycbcr(image2).astype('uint8')[:,:,0]/255.0
	im_gt_ycbcr = cropping(im_gt_ycbcr,height,width)
	image2_raw = im_gt_ycbcr
	#print(image2.shape)
	print(image2_raw)

	if os.path.exists(result_path_dir):
	files = [f for f in os.listdir(result_path_dir)]
	for f in files:
		os.remove(str(result_path_dir+f))
	else:
	os.makedirs(result_path_dir)


for i, item in enumerate(file_list):
  file_name = os.path.splitext(item)
  img_raw = imread(DATA_PATH+item)
  width = img_raw.shape[1]
  height = img_raw.shape[0]
  if len(img_raw.shape) == 3:
    im_gt_ycbcr = np.round(rgb2ycbcr(img_raw)).astype('uint8')
    im_gt_ycbcr = cropping(im_gt_ycbcr,height,width)
    img_raw = im_gt_ycbcr[:,:,0]/255.0
    is_gray = False
  elif len(img_raw.shape) == 2:
    im_gt_ycbcr = 0
    img_raw = img_raw/255.0
    is_gray = True
print("VDSR Success")
'''
