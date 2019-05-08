from __future__ import print_function, absolute_import

import os
import argparse
import time
import matplotlib.pyplot as plt

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.datasets as datasets

import _init_paths
from pytorch-pose.pose import Bar
from pytorch-pose.pose.utils.logger import Logger, savefig
from pytorch-pose.pose.utils.evaluation import accuracy, AverageMeter, final_preds

from pytorch-pose.pose.utils.misc import save_checkpoint, save_pred, adjust_learning_rate
from pytorch-pose.pose.utils.osutils import mkdir_p, isfile, isdir, join
from pytorch-pose.pose.utils.imutils import batch_with_heatmap
from pytorch-pose.pose.utils.transforms import fliplr, flip_back
from pytorch-pose.pose import models as models
from pytorch-pose.pose import datasets as datasets
from pytorch-pose.pose import losses as losses

import cv2, nonechucks as nc, numpy as np 

class Hourglass():

	def __init__(self):

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		cudnn.benchmark = True

		self.img_path=''
		self.dataset = 'mpii'
		self.image_path = ''
		self.inp_res = 256
		self.out_res = 64

		self.arch = 'hg'
		self.stacks = 1
		self.blocks = 1
		self.features = 256
		self.resnet_layers = 50

		self.solver = 'rms'
		self.workers = 1
		self.epochs = 100
		self.test_batch=1
		self.train_batch=1
		self.lr = 2.5e-4
		self.momentum=0
		self.weight_decay=0
		self.gamma=0.1

		self.sigma=1.0
		self.scale_factor=0.25
		self.rot_factor=1
		self.sigma_decay=0

		#self.checkpoint=''
		self.resume=''
		self.njoints=24

		self.model = models.self.dataset(num_stacks=self.stacks, num_blocks=self.blocks, num_classes=self.njoints, resnet_layers=self.resnet_layers)

		self.model = torch.nn.DataParallel(self.model).to(self.device)
		self.criterion = losses.JointsMSELoss().to(self.device)
		self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)

		self.checkpoint = torch.load("/home/shantam/Documents/Programs/pytorch-pose/checkpoint/mpii/hg_fullset/checkpoint.pth.tar")
		self.start_epoch = checkpoint['epoch']
		self.model.load_state_dict(self.checkpoint['state_dict'])
		self.model.eval()

	def forward_pass(self, img):

		points = []
		pointers = []

		c = [img.shape[1]/2, img.shape[2]/2]
		s = float(img.shape[1]/200.0)
	
		img = crop(self.img_path, img, c, s, [self.inp_res, self.inp_res])
		img = img.to(device, non_blocking = True)
		output=self.model(img)
	
		score_map = output[-1].cpu() if type(output) == list else output.cpu()
		preds, vals = final_preds(score_map, [c], [s], [64, 64])
		coords = np.squeeze(preds)

		for m in range(0,len(coords)):
			val = vals[0][m].detach().numpy()
			if val>0.25: #threshold for confidence score
				x,y = coords[m][0].cpu().detach().numpy(), coords[m][1].cpu().detach().numpy()
				pointers.append([x,y])
				points.append(m)
		
		return points, pointers

