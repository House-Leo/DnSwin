import os
import random
import torch
import numpy as np
import glob
from torch.utils.data import Dataset

from utils import read_img, hwc_to_chw


def get_patch(clean_img, noise_img, patch_size):
	H = clean_img.shape[0]
	W = clean_img.shape[1]

	ps_temp = min(H, W, patch_size)

	xx = np.random.randint(0, W-ps_temp) if W > ps_temp else 0
	yy = np.random.randint(0, H-ps_temp) if H > ps_temp else 0

	patch_clean_img = clean_img[yy:yy+ps_temp, xx:xx+ps_temp, :].copy()
	patch_noise_img = noise_img[yy:yy+ps_temp, xx:xx+ps_temp, :].copy()

	if np.random.randint(2, size=1)[0] == 1:
		patch_clean_img = np.flip(patch_clean_img, axis=1)
		patch_noise_img = np.flip(patch_noise_img, axis=1)
	if np.random.randint(2, size=1)[0] == 1: 
		patch_clean_img = np.flip(patch_clean_img, axis=0)
		patch_noise_img = np.flip(patch_noise_img, axis=0)
	if np.random.randint(2, size=1)[0] == 1:
		patch_clean_img = np.transpose(patch_clean_img, (1, 0, 2))
		patch_noise_img = np.transpose(patch_noise_img, (1, 0, 2))

	return patch_clean_img, patch_noise_img


class Base(Dataset):
	def __init__(self, root_dir, sample_num, patch_size=0):
		self.patch_size = patch_size

		folders = sorted(glob.glob(root_dir + '/*'))

		self.clean_fns = [None] * sample_num
		for i in range(sample_num):
			self.clean_fns[i] = []

		for ind, folder in enumerate(folders):
			clean_imgs = sorted(glob.glob(folder + '/*GT_SRGB*'))

			for clean_img in clean_imgs:
				self.clean_fns[ind % sample_num].append(clean_img)

	def __len__(self):
		l = len(self.clean_fns)
		return l

	def __getitem__(self, idx):
		clean_fn = random.choice(self.clean_fns[idx])

		clean_img = read_img(clean_fn)
		noise_img = read_img(clean_fn.replace('GT_SRGB', 'NOISY_SRGB'))

		if self.patch_size > 0:
			clean_img, noise_img = get_patch(clean_img, noise_img, self.patch_size)

		clean_img_chw = hwc_to_chw(clean_img)
		noise_img_chw = hwc_to_chw(noise_img)

		return noise_img_chw, clean_img_chw