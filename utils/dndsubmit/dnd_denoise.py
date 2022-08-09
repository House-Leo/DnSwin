 # Author: Tobias Plötz, TU Darmstadt (tobias.ploetz@visinf.tu-darmstadt.de)

 # This file is part of the implementation as described in the CVPR 2017 paper:
 # Tobias Plötz and Stefan Roth, Benchmarking Denoising Algorithms with Real Photographs.
 # Please see the file LICENSE.txt for the license governing this code.

import imageio
import numpy as np
import scipy.io as sio
import os
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F

from skimage import io
from tqdm import tqdm

from ..common import data_augmentation, inverse_data_augmentation


def denoise_srgb(model, data_folder, out_folder, ensemble=False):
    try:
        os.makedirs(out_folder)
    except:pass

    patch_size = 512
    padding = 16 if ensemble else 0
    anum = 8 if ensemble else 1

    print('model loaded\n')
    # load info
    infos = h5py.File(os.path.join(data_folder, 'info.mat'), 'r')
    info = infos['info']
    bb = info['boundingboxes']
    print('info loaded\n')
    # process data
    for i in tqdm(range(50)):
        filename = os.path.join(data_folder, 'images_srgb', '%04d.mat'%(i+1))
        img = h5py.File(filename, 'r')
        Inoisy = np.float32(np.array(img['InoisySRGB']).T)
        # bounding box
        ref = bb[0][i]
        boxes = np.array(info[ref]).T
        for k in range(20):
            idx = [int(boxes[k,0]-1),int(boxes[k,2]),int(boxes[k,1]-1),int(boxes[k,3])]
            Inoisy_crop = Inoisy[idx[0]-padding:idx[1]+padding,idx[2]-padding:idx[3]+padding,:].copy()

            Inoisy_list = []
            Idenoised_list = []
            for ir in range(anum):
                Inoisy_crop_t = np.ascontiguousarray(data_augmentation(Inoisy_crop, ir))
                Inoisy_crop_tensor = torch.from_numpy(np.transpose(Inoisy_crop_t, axes=[2, 0, 1])).unsqueeze(0).cuda()
                Inoisy_list.append(Inoisy_crop_tensor)

            input_tensor = torch.cat(Inoisy_list, dim=0)
            output_tensor = model(input_tensor)

            for ir in range(anum):
                denoised_t = np.transpose(output_tensor[ir,...].cpu().numpy(), [1,2,0])
                Idenoised_crop_t = inverse_data_augmentation(denoised_t, ir)
                Idenoised_list.append(Idenoised_crop_t)
            Idenoised_crop = np.clip(sum(Idenoised_list) / anum, 0, 1)

            # save denoised data
            Idenoised_crop = np.float32(Idenoised_crop)[padding:patch_size+padding, padding:patch_size+padding, :]
            save_file = os.path.join(out_folder, '%04d_%02d.mat'%(i+1,k+1))
            sio.savemat(save_file, {'Idenoised_crop': Idenoised_crop})

            print('%s crop %d/%d' % (filename, k+1, 20))
        print('[%d/%d] %s done\n' % (i+1, 50, filename))