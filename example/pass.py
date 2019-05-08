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
import torchvision.transforms as transforms

import _init_paths
from pose import Bar
from pose.utils.logger import Logger, savefig
from pose.utils.evaluation import accuracy, AverageMeter, final_preds

from pose.utils.misc import save_checkpoint, save_pred, adjust_learning_rate
import pose.models as models
import pose.datasets as datasets
import pose.losses as losses

from pose.utils.osutils import *
from pose.utils.imutils import *
from pose.utils.transforms import *

import cv2, nonechucks as nc, numpy as np, random

# from PIL import Image
# from pathlib import Path

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


dataset_names = sorted(name for name in datasets.__dict__
    if name.islower() and not name.startswith("__")
    and callable(datasets.__dict__[name]))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True

def main(args):

	img_path = "/home/shantam/Documents/Programs/hourglasstensorlfow/images/cropped0.jpg"
	img = load_image(img_path)
	c = [img.shape[1]/2, img.shape[2]/2]
	s = float(img.shape[1]/200.0)
	
	img = crop(img_path, img, c, s, [256, 256])
	trans = transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
	#img = trans(img)
	img = img.unsqueeze(0)
	print (img.shape)

	njoints=24

	model = models.__dict__[args.arch](num_stacks=args.stacks,
                                       num_blocks=args.blocks,
                                       num_classes=njoints,
                                       resnet_layers=args.resnet_layers)

	model = torch.nn.DataParallel(model).to(device)
	criterion = losses.JointsMSELoss().to(device)

	if args.solver == 'rms':
		print ("done")
		optimizer = torch.optim.RMSprop(model.parameters(),
                                        lr=args.lr,
                                        momentum=args.momentum,
                                        weight_decay=args.weight_decay)
	elif args.solver == 'adam':
		optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
        )

	checkpoint = torch.load("/home/shantam/Documents/Programs/pytorch-pose/checkpoint/mpii/hg_fullset/checkpoint.pth.tar")
	args.start_epoch = checkpoint['epoch']
	model.load_state_dict(checkpoint['state_dict'])
	model.eval()
	img = img.to(device, non_blocking = True)
	output=model(img)

	#print (len(output))

	score_map = output[-1].cpu() if type(output) == list else output.cpu()
	preds, vals = final_preds(score_map, [c], [s], [64, 64])
	
	image=cv2.imread(img_path)
	coords = np.squeeze(preds)

	for m in range(0,len(coords)):
		val = vals[0][m].detach().numpy()
		print (val)
		if val>0.25: #threshold for confidence score
			x,y = coords[m][0].cpu().detach().numpy(), coords[m][1].cpu().detach().numpy()
			print (x,y)
			cv2.circle(image, (x,y), 1, (0,0,255), -1)

	while True:
		cv2.imshow("dec", image)
		cv2.waitKey(10)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    # Dataset setting
    parser.add_argument('--dataset', metavar='DATASET', default='mpii',
                        choices=dataset_names,
                        help='Datasets: ' +
                            ' | '.join(dataset_names) +
                            ' (default: mpii)')
    parser.add_argument('--image-path', default='', type=str,
                        help='path to images')
    parser.add_argument('--anno-path', default='', type=str,
                        help='path to annotation (json)')
    parser.add_argument('--year', default=2014, type=int, metavar='N',
                        help='year of coco dataset: 2014 (default) | 2017)')
    parser.add_argument('--inp-res', default=256, type=int,
                        help='input resolution (default: 256)')
    parser.add_argument('--out-res', default=64, type=int,
                    help='output resolution (default: 64, to gen GT)')

    # Model structure
    parser.add_argument('--arch', '-a', metavar='ARCH', default='hg',
                        choices=model_names,
                        help='model architecture: ' +
                            ' | '.join(model_names) +
                            ' (default: hg)')
    parser.add_argument('-s', '--stacks', default=1, type=int, metavar='N',
                        help='Number of hourglasses to stack')
    parser.add_argument('--features', default=256, type=int, metavar='N',
                        help='Number of features in the hourglass')
    parser.add_argument('--resnet-layers', default=50, type=int, metavar='N',
                        help='Number of resnet layers',
                        choices=[18, 34, 50, 101, 152])
    parser.add_argument('-b', '--blocks', default=1, type=int, metavar='N',
                        help='Number of residual modules at each location in the hourglass')
    # Training strategy
    parser.add_argument('--solver', metavar='SOLVER', default='rms',
                        choices=['rms', 'adam'],
                        help='optimizers')
    parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--train-batch', default=6, type=int, metavar='N',
                        help='train batchsize')
    parser.add_argument('--test-batch', default=1, type=int, metavar='N',
                        help='test batchsize')
    parser.add_argument('--lr', '--learning-rate', default=2.5e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                        metavar='W', help='weight decay (default: 0)')
    parser.add_argument('--schedule', type=int, nargs='+', default=[60, 90],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='LR is multiplied by gamma on schedule.')
    parser.add_argument('--target-weight', dest='target_weight',
                        action='store_true',
                        help='Loss with target_weight')
    # Data processing
    parser.add_argument('-f', '--flip', dest='flip', action='store_true',
                        help='flip the input during validation')
    parser.add_argument('--sigma', type=float, default=1,
                        help='Groundtruth Gaussian sigma.')
    parser.add_argument('--scale-factor', type=float, default=0.25,
                        help='Scale factor (data aug).')
    parser.add_argument('--rot-factor', type=float, default=1,
                        help='Rotation factor (data aug).')
    parser.add_argument('--sigma-decay', type=float, default=0,
                        help='Sigma decay rate for each epoch.')
    parser.add_argument('--label-type', metavar='LABELTYPE', default='Gaussian',
                        choices=['Gaussian', 'Cauchy'],
                        help='Labelmap dist type: (default=Gaussian)')
    # Miscs
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--snapshot', default=0, type=int,
                        help='save models for every #snapshot epochs (default: 0)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('-d', '--debug', dest='debug', action='store_true',
                        help='show intermediate results')


    main(parser.parse_args())