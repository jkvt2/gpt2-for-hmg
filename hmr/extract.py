"""
Demo of HMR.

Note that HMR requires the bounding box of the person in the image. The best performance is obtained when max length of the person in the image is roughly 150px. 

When only the image path is supplied, it assumes that the image is centered on a person whose length is roughly 150px.
Alternatively, you can supply output of the openpose to figure out the bbox and the right scale factor.

Sample usage:

# On images on a tightly cropped image around the person
python -m demo --img_path data/im1963.jpg
python -m demo --img_path data/coco1.png

# On images, with openpose output
python -m demo --img_path data/random.jpg --json_path data/random_keypoints.json
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from absl import flags
import numpy as np

import skimage.io as io
import tensorflow as tf

from src.util import image as img_util
from src.util import openpose as op_util
import src.config
from src.RunModel import RunModel
import pickle
import os

def preprocess_image(img_path, json_path=None):
    img = io.imread(img_path)
    if img.shape[2] == 4:
        img = img[:, :, :3]

    if json_path is None:
        if np.max(img.shape[:2]) != config.img_size:
            print('Resizing so the max image size is %d..' % config.img_size)
            scale = (float(config.img_size) / np.max(img.shape[:2]))
        else:
            scale = 1.
        center = np.round(np.array(img.shape[:2]) / 2).astype(int)
        # image center in (x,y)
        center = center[::-1]
    else:
        scale, center = op_util.get_bbox(json_path)

    crop, proc_param = img_util.scale_and_crop(img, scale, center,
                                               config.img_size)

    # Normalize image to [-1, 1]
    crop = 2 * ((crop / 255.) - 0.5)

    return crop, proc_param, img


def main():
    sess = tf.Session()
    model = RunModel(config, sess=sess)

    for walk_type in sorted(os.listdir('../data/preproc')):
        thetas = []
        outfn = '../data/hmr/{}.pkl'.format(walk_type)
        if os.path.exists(outfn):continue
        for fn in sorted(os.listdir(os.path.join('../data/preproc', walk_type))):
            img_path = os.path.join('../data/preproc', walk_type, fn)
            jfn = fn.split('.')[0] + '_keypoints.json'
            json_path = os.path.join('../data/json', walk_type, jfn)
            
            input_img, proc_param, img = preprocess_image(img_path, json_path)
            # Add batch dimension: 1 x D x D x 3
            input_img = np.expand_dims(input_img, 0)
        
            # Theta is the 85D vector holding [camera, pose, shape]
            # where camera is 3D [s, tx, ty]
            # pose is 72D vector holding the rotation of 24 joints of SMPL in axis angle format
            # shape is 10D shape coefficients of SMPL
            joints, verts, cams, joints3d, theta = model.predict(
                input_img, get_theta=True)
            thetas += [theta[0]]
        with open(outfn, 'wb') as f:
            pickle.dump(np.stack(thetas), f)

if __name__ == '__main__':
    config = flags.FLAGS
    config(sys.argv)
    # Using pre-trained model, change this to use your own.
    config.load_path = src.config.PRETRAINED_MODEL

    config.batch_size = 1

    main()