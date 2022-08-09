import os, shutil, importlib
import torch
import torch.nn as nn
import argparse
import numpy as np
import scipy.io as sio
import h5py
import glob
import cv2
from tqdm import tqdm


def hwc_to_chw(img):
    return np.transpose(img, axes=[2, 0, 1]).astype('float32')
	

def data_augmentation(image, mode):
    '''
    Performs data augmentation of the input image
    Input:
        image: a cv2 (OpenCV) image
        mode: int. Choice of transformation to apply to the image
                0 - no transformation
                1 - flip up and down
                2 - rotate counterwise 90 degree
                3 - rotate 90 degree and flip up and down
                4 - rotate 180 degree
                5 - rotate 180 degree and flip
                6 - rotate 270 degree
                7 - rotate 270 degree and flip
    '''
    if mode == 0:
        # original
        out = image
    elif mode == 1:
        # flip up and down
        out = np.flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(image)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(image)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(image, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(image, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(image, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(image, k=3)
        out = np.flipud(out)
    else:
        raise Exception('Invalid choice of image transformation')

    return out


def inverse_data_augmentation(image, mode):
    '''
    Performs inverse data augmentation of the input image
    '''
    if mode == 0:
        # original
        out = image
    elif mode == 1:
        out = np.flipud(image)
    elif mode == 2:
        out = np.rot90(image, axes=(1,0))
    elif mode == 3:
        out = np.flipud(image)
        out = np.rot90(out, axes=(1,0))
    elif mode == 4:
        out = np.rot90(image, k=2, axes=(1,0))
    elif mode == 5:
        out = np.flipud(image)
        out = np.rot90(out, k=2, axes=(1,0))
    elif mode == 6:
        out = np.rot90(image, k=3, axes=(1,0))
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.flipud(image)
        out = np.rot90(out, k=3, axes=(1,0))
    else:
        raise Exception('Invalid choice of image transformation')

    return out

parser = argparse.ArgumentParser(description = 'Test')
parser.add_argument('--model', default='DnSwin', type=str, help='model name')
args = parser.parse_args()

input_dir = './data'
output_dir = './data'
save_dir = os.path.join('./save_model/', args.model)

# noisy_filename = os.path.join(input_dir, 'BenchmarkNoisyBlocksSrgb.mat')
denoised_filename = os.path.join(output_dir, 'SubmitSrgb.mat')
bat_denoised_filename = os.path.join(output_dir, 'SubmitSrgb_bat.mat')

if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

shutil.copyfile(denoised_filename, bat_denoised_filename)

'''model here'''
model = importlib.import_module('.' + args.model, package='model').Network()
model.cuda()
model = nn.DataParallel(model)

model.eval()

if os.path.exists(os.path.join(save_dir, 'best_model.pth.tar')):
    # load existing model
    model_info = torch.load(os.path.join(save_dir, 'best_model.pth.tar'))
    model.load_state_dict(model_info['state_dict'])

'''two methods to load noisy patch'''
# noisy_filename = os.path.join(input_dir, 'BenchmarkNoisyBlocks', 'BenchmarkNoisyBlocksSrgb.mat')
# noisy_imgs = sio.loadmat(noisy_filename)['BenchmarkNoisyBlocksSrgb']

Blocks32 = sio.loadmat(os.path.join(input_dir, 'BenchmarkBlocks32.mat'))['BenchmarkBlocks32']

full_img_list = glob.glob(os.path.join(input_dir, 'SIDD_Benchmark_Data/*/*.PNG'))
full_img_list.sort()

with h5py.File(denoised_filename, 'r+') as denoised_file:
    denoised_imgs = np.array(denoised_file['DenoisedBlocksSrgb']).T

    N, B = denoised_imgs.shape

    for i in tqdm(range(N)):
        full_img = cv2.imread(full_img_list[i])[:,:,::-1]

        for j in range(B):
            yy = Blocks32[j][0] - 1
            xx = Blocks32[j][1] - 1
            ps = 256            # Blocks[2] == Blocks[3]
            pad = 32

            noisy_img = full_img[yy-pad:yy+ps+pad, xx-pad:xx+ps+pad, :] #(H=2*pad+ps,W=2*pad+ps,C)

            output_nps = []

            in_img_list = []
            for t in range(8):
                in_img = noisy_img.copy()
                in_img = data_augmentation(in_img, t) #翻转做数据增强再输入
                in_img_list.append(torch.from_numpy(hwc_to_chw(np.float32(in_img) / 255.)).cuda().unsqueeze_(0))

            input_var = torch.cat(in_img_list, dim=0)

            with torch.no_grad():
                if args.model == 'gmsnet':
                    output = model(input_var)
                else:
                    output = model(input_var)                

            out_img_list = []
            for t in range(8):
                out_img_list.append(inverse_data_augmentation(np.transpose(output[t,...].cpu().numpy(), [1,2,0]), t)) #输出后再逆翻转恢复原来图片

            denoised_img = np.round((np.clip(sum(out_img_list) / 8, 0, 1) * 255.)).astype('uint8')[pad:pad+ps, pad:pad+ps, :] # /8取平均 再取pad:pad+ps的框，相当于做ensemble

            # save to mat
            img_ref = denoised_file[denoised_imgs[i, j]]
            img_ref[...] = denoised_img.T

            # out_folder = os.path.join(output_dir, '%04d'%(i+321))

            # if not os.path.isdir(out_folder):
            #     os.makedirs(out_folder)

            # io.imsave(os.path.join(out_folder, 'GT_SRGB_%d.png' % j), denoised_img)
            # io.imsave(os.path.join(out_folder, 'NOISY_SRGB_%d.png' % j), noisy_img[pad:pad+ps, pad:pad+ps, :])