import os
import pickle
from hmr.src.tf_smpl.batch_smpl import SMPL
from hmr.src.util import renderer as vis_util
from hmr.src.util import image as img_util
from hmr.src.util import openpose as op_util
from hmr.src.tf_smpl import projection as proj_util
import skimage.io as io
# import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from GPT2.data_utils import load_all_examples, get_data_split, get_norm_params

SMPL_MODEL_PATH = os.path.join(
    'hmr', 'models', 'neutral_smpl_with_cocoplus_reg.pkl')
SMPL_FACE_PATH = os.path.join(
    'hmr', 'src', 'tf_smpl', 'smpl_faces.npy')

def preprocess_image(img_path, json_path=None, img_size=256):
    img = io.imread(img_path)
    if img.shape[2] == 4:
        img = img[:, :, :3]

    if json_path is None:
        if np.max(img.shape[:2]) != img_size:
            print('Resizing so the max image size is %d..' % img_size)
            scale = (float(img_size) / np.max(img.shape[:2]))
        else:
            scale = 1.
        center = np.round(np.array(img.shape[:2]) / 2).astype(int)
        # image center in (x,y)
        center = center[::-1]
    else:
        scale, center = op_util.get_bbox(json_path)

    crop, proc_param = img_util.scale_and_crop(img, scale, center,
                                               img_size)

    # Normalize image to [-1, 1]
    crop = 2 * ((crop / 255.) - 0.5)

    return crop, proc_param, img

class Visualiser():
    def __init__(self,):
        tf.reset_default_graph()
        self.smpl = SMPL(SMPL_MODEL_PATH, joint_type='cocoplus')
        self.renderer = vis_util.SMPLRenderer(face_path=SMPL_FACE_PATH)
        self.proj_fn = proj_util.batch_orth_proj_idrot
    
    def render(self, theta, proc_param, img_size=(256, 256)):
        cams = theta[None, :3]
        poses = theta[None, 3:3+72]
        shapes = theta[None, 3+72:]
        verts, Js, _ = self.smpl(tf.constant(shapes), tf.constant(poses), get_skin=True)
        verts, Js = self.sess.run([verts, Js])
        joints = self.proj_fn(Js, cams)
        cam_for_render, vert_shifted, kp_original = \
          vis_util.get_original(proc_param, verts[0], cams[0], joints[0], 1080)
        
        rend_img = self.renderer(
            vert_shifted, cam=cam_for_render, img_size=img_size)
        return rend_img
    
    def __enter__(self):
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.sess.close()

def main_hmr(json=True):
    pkl_dir = os.path.join('data', 'hmr')
    vis_dir = os.path.join('data', 'hmr_vis')
    img_dir = os.path.join('data', 'preproc')
    json_dir = os.path.join('data', 'json')
    
    for pkl in sorted(os.listdir(pkl_dir)):
        with Visualiser() as vis:
            with open(os.path.join(pkl_dir, pkl), 'rb') as f:
                thetas = pickle.load(f)
            walk_type = pkl.split('.')[0]
            if os.path.exists(os.path.join(vis_dir, walk_type)):continue
            os.makedirs(os.path.join(vis_dir, walk_type), exist_ok=True)
            for i, theta in enumerate(thetas):
                img_path = os.path.join(img_dir, walk_type, 'out{:03d}.png'.format(i))
                json_path = os.path.join(json_dir, walk_type, 'out{:03d}_keypoints.json'.format(i))
                _, proc_param, _ = preprocess_image(
                    img_path=img_path,
                    json_path=json_path if json else None)
                rend_img = vis.render(theta, proc_param)
                io.imsave(os.path.join(vis_dir, walk_type, 'out{:03d}.png'.format(i)),
                      rend_img)

def main_gpt():
    trn_ex, _ = load_all_examples(
        get_data_split(
            data_split_file='data/split',
            hmr_dir='data/hmr'),
        hmr_dir='data/hmr')
    mean, std = get_norm_params(trn_ex)
    proc_param = {
        'scale': 1.0,
        'start_pt': np.array([128, 128]),
        'end_pt': np.array([384, 384]),
        'img_size': 256}
    for dataset in ['test', 'train']:
        pkl_dir = os.path.join('GPT2', 'logs', dataset)
        vis_dir = os.path.join('data', 'gpt_vis', dataset)
        for f in os.listdir(pkl_dir):
            if not f.endswith('.pkl'):continue
            pkl_fn = os.path.join(pkl_dir, f)
            walk_type = f.split('.')[0]
            if os.path.exists(os.path.join(vis_dir, walk_type)): continue
            os.makedirs(os.path.join(vis_dir, walk_type))
            with Visualiser() as vis:
                with open(pkl_fn, 'rb') as f:
                    thetas = pickle.load(f)
                for i, theta in enumerate(thetas):
                    if i <19:continue
                    rend_img = vis.render(theta * std + mean, proc_param)
                    io.imsave(os.path.join(vis_dir, walk_type, 'out{:03d}.png'.format(i)),
                          rend_img)

if __name__ == '__main__':
    # main_hmr(json=False)
    main_gpt()