import os
import skimage.io as io
import skimage.transform
import numpy as np

#sort
starting_frame = 73
num_frames_per = 90
with open('styles.csv', 'r') as f:
    walk_types = f.read().rstrip().split(',')
for i, wt in enumerate(walk_types):
    print(wt)
    out_dir = os.path.join(
        'preproc',
        '{:02d}_{}'.format(i, wt))
    os.makedirs(out_dir, exist_ok=True)
    this_starting_frame = starting_frame+i*num_frames_per
    ids = ['out{}.png'.format(j) for j in range(
        this_starting_frame,
        this_starting_frame + num_frames_per,)]
    for i, fn in enumerate(ids):
        src = os.path.join('raw', fn)
        dst = os.path.join(out_dir, 'out{:03d}.png'.format(i))
        if os.path.exists(dst):continue
        img = io.imread(src)
        x = skimage.transform.resize(img, (256, 256), preserve_range=True).astype(np.uint8)
        io.imsave(dst, x)