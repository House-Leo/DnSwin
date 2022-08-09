import os, importlib
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from losses import CharbonnierLoss
from torch.utils.tensorboard import SummaryWriter
from utils import AverageMeter
from dataset.loader import Base

parser = argparse.ArgumentParser(description = 'Train')
parser.add_argument('--model', default='DnSwin', type=str, help='model name')
parser.add_argument('--bs', default=16, type=int, help='batch size')
parser.add_argument('--ps', default=128, type=int, help='patch size')
parser.add_argument('--nw', default=0, type=int, help='number of workers')
parser.add_argument('--lr', default=2e-4, type=float, help='learning rate')
parser.add_argument('--epochs', default=6000, type=int, help='sum of epochs')
parser.add_argument('--eval_freq', default=100, type=int, help='evaluation frequency')
parser.add_argument('--gpus', default='2', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--resume', action='store_true', help='resume to train')
parser.add_argument('--pre', action='store_true', help='pretrain_model to train')
parser.add_argument('--pre_model', default='./save_model/DnSwin/', type=str, help='resume to train')
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

def train(loader1, loader2, model, criterion, optimizer):
	losses = AverageMeter()
	model.train()

	for (pairs1, pairs2) in tqdm(zip(loader1, loader2), total=min(len(loader1), len(loader2))):
		noise_img = torch.cat([pairs1[0], pairs2[0]], dim=0)
		clean_img = torch.cat([pairs1[1], pairs2[1]], dim=0)

		input_var = noise_img.cuda()
		target_var = clean_img.cuda()

		output = model(input_var)
		loss = criterion(output, target_var)
		losses.update(loss.item())

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
	
	return losses.avg

def valid(loader, model, criterion):
	losses = AverageMeter()
	model.train()

	for (noise_img, clean_img) in tqdm(loader):
		input_var = noise_img.cuda()
		target_var = clean_img.cuda()

		with torch.no_grad():
			output = model(input_var)

		loss = criterion(output, target_var)
		losses.update(loss.item())

	return losses.avg


if __name__ == '__main__':
	save_dir = os.path.join('./save_model/', args.model)
	logs_dir = os.path.join('./log/', args.model)
	
	model = importlib.import_module('.' + args.model, package='model').Network()
	model.cuda()

	device_ids = [i for i in range(torch.cuda.device_count())]
	if torch.cuda.device_count() > 1:
  		print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")
	if len(device_ids)>1:
		model = nn.DataParallel(model, device_ids = device_ids)
	if args.pre:
		model_info = torch.load(os.path.join(args.pre_model, 'best_model.pth.tar'))
		print('==> loading existing model:', os.path.join(args.pre_model, 'best_model.pth.tar'))
		model.load_state_dict(model_info['state_dict'], strict=False)
	if args.resume:
		if os.path.exists(os.path.join(save_dir, 'checkpoint.pth.tar')):
			# load existing model
			model_info = torch.load(os.path.join(save_dir, 'checkpoint.pth.tar'))
			print('==> loading existing model:', os.path.join(save_dir, 'checkpoint.pth.tar'))
			model.load_state_dict(model_info['state_dict'])
			optimizer = torch.optim.Adam(model.parameters())
			optimizer.load_state_dict(model_info['optimizer'])
			scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
			scheduler.load_state_dict(model_info['scheduler'])
			cur_epoch = model_info['epoch']
			best_loss = model_info['loss']
	else:
		if not os.path.isdir(save_dir):
			os.makedirs(save_dir)
		if not os.path.isdir(logs_dir):
			os.makedirs(logs_dir)
		# create model
		optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-8)
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
		cur_epoch = 0
		best_loss = 10.0
		
	criterion = CharbonnierLoss()
	# criterion = nn.L1Loss()
	criterion.cuda()

	train_dataset1 = Base('./data/SIDD_train/', 320, args.ps)
	train_loader1 = torch.utils.data.DataLoader(
		train_dataset1, batch_size=(args.bs-(args.bs//4)), shuffle=True, num_workers=args.nw, pin_memory=True, drop_last=True)

	train_dataset2 = Base('./data/Syn_train/', 100, args.ps)
	train_loader2 = torch.utils.data.DataLoader(
		train_dataset2, batch_size=(args.bs//4), shuffle=True, num_workers=args.nw, pin_memory=True, drop_last=True)

	valid_dataset = Base('./data/SIDD_valid/', 1280)
	valid_loader = torch.utils.data.DataLoader(
		valid_dataset, batch_size=args.bs, num_workers=args.nw, pin_memory=True)
	
	writer = SummaryWriter(logs_dir)
	for epoch in range(cur_epoch, args.epochs + 1):
		if epoch > 6000:
			args.eval_freq = 50
		train_loss = train(train_loader1, train_loader2, model, criterion, optimizer)
		scheduler.step()
		if epoch % args.eval_freq == 0:
			avg_loss = valid(valid_loader, model, criterion)
			writer.add_scalar('Val Loss', avg_loss, epoch)

			if avg_loss < best_loss:
				best_loss = avg_loss
				torch.save({
					'epoch': epoch + 1,
					'loss': best_loss,
					'state_dict': model.state_dict(),
					'optimizer' : optimizer.state_dict(),
					'scheduler' : scheduler.state_dict()}, 
					os.path.join(save_dir, 'best_model.pth.tar'))
			writer.add_scalar('Best Loss', best_loss, epoch)

		torch.save({
			'epoch': epoch + 1,
			'loss': best_loss,
			'state_dict': model.state_dict(),
			'optimizer' : optimizer.state_dict(),
			'scheduler' : scheduler.state_dict()}, 
			os.path.join(save_dir, 'checkpoint.pth.tar'))

		print('Epoch [{0}]\t'
			'lr: {lr:.6f}\t'
			'Train Loss: {train_loss:.5f}\t'
			'Best valid loss: {valid_loss:.5f}'
			.format(
			epoch,
			lr=optimizer.param_groups[-1]['lr'],
			train_loss=train_loss,
			valid_loss=best_loss))
		
		writer.add_scalar('Train Loss', train_loss, epoch)
		
							