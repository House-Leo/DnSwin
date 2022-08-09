import os, importlib
import torch
import torch.nn as nn
import argparse

from utils import denoise_srgb, bundle_submissions_srgb

parser = argparse.ArgumentParser(description = 'Test')
parser.add_argument('--model', default='DnSwin', type=str, help='model name')
parser.add_argument('--ensemble', action='store_true', default=False, help='ensemble result')
args = parser.parse_args()

input_dir = './data/DND/'
output_dir = './result/submit_dnd/'
save_dir = os.path.join('./save_model/', args.model)

model = importlib.import_module('.' + args.model, package='model').Network()

model.cuda()
model = nn.DataParallel(model)


model.eval()

if os.path.exists(os.path.join(save_dir, 'best_model.pth.tar')):
    # load existing model
    model_info = torch.load(os.path.join(save_dir, 'best_model.pth.tar'))
    model.load_state_dict(model_info['state_dict'])
else:
    print('Error: no trained model detected!')
    exit(1)

with torch.no_grad():
    denoise_srgb(model, input_dir, output_dir, args.ensemble)

bundle_submissions_srgb(output_dir)