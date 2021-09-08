from __future__ import print_function
import torch.backends.cudnn as cudnn
from utils import *
from models.utils import get_model
from passers import Passer
from loaders import *
from graph import *
from labels import load_manipulator
from sklearn.manifold import MDS

import numpy as np
import os

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--net')
parser.add_argument('--dataset')
parser.add_argument('--save_path')
parser.add_argument('--trial', default=0, type=int)
parser.add_argument('--epochs', nargs='+', type=int)
parser.add_argument('--split', default=0, type=int)
parser.add_argument('--kl', default=0, type=int)
parser.add_argument('--input_size', default=32, type=int)
parser.add_argument('--filtration', default='nominal')
parser.add_argument('--permute_labels', default=0, type=float)
parser.add_argument('--binarize_labels', default=-1, type=int)
parser.add_argument('--playground_layers', nargs='+', type=int)
parser.add_argument('--dim', type=int, default=4)
parser.add_argument('--row_count', type=int, default=1024)

args = parser.parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
''' Meta-name to be used as prefix on all savings'''
SAVE_DIR = os.path.join(args.save_path, args.net + '_' + args.dataset + '/')
START_LAYER = 3 if args.net in ['vgg', 'resnet'] else 0 

''' If save directory doesn't exist create '''
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)    

# Build models
print('==> Building model..')
net = get_model(args.net, args.dataset, args.playground_layers)
net = net.to(device)
    
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

print(net)

''' Prepare criterion '''
criterion = nn.CrossEntropyLoss()

''' Define label manipulator '''
manipulator = load_manipulator(args.permute_labels, args.binarize_labels)
    
for epoch in args.epochs:
    print('==> Loading checkpoint for epoch {}...'.format(epoch))
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/'+ args.net + '_' + args.dataset + '/ckpt_trial_' + str(args.trial) + '_epoch_' + str(epoch)+'.t7')
    net.load_state_dict(checkpoint['net'])
    
    ''' Define passer and get activations '''
    functloader = loader(args.dataset+'_test', batch_size=100, subset=list(range(0, 1000)), row_count = args.row_count, dim=args.dim)
    passer = Passer(net, functloader, criterion, device)
    passer_test = Passer(net, functloader, criterion, device)
    passer_test.run(manipulator=manipulator)
    activs = passer.get_function()
    activs = signal_concat(activs)

    if not os.path.exists("plot"):
        os.makedirs("plot")

    np.save(os.path.join("plot", f"activations_epoch_{epoch}"), np.array(activs))
    print(f"Activations saved as activations_epoch_{epoch}")

    # What follows performs a MDS on the activation values to see how it impacts homology, done very badly


    #embedding = MDS(n_components=3)
    #activs = embedding.fit_transform(activs.astype(np.double))

    adj = adjacency(activs)

    print('The dimension of the adjacency matrix is {}'.format(adj.shape))
    print('Adj mean {}, min {}, max {}'.format(np.mean(adj), np.min(adj), np.max(adj)))

    np.save(os.path.join("plot", f"distances_epoch_{epoch}"), adj)
    print(f"Distances saved as distances_epoch_{epoch}")



    ''' Write adjacency to binary. To use as DIPHA input for persistence homology '''
    save_dipha(SAVE_DIR + 'adj_epc{}_trl{}.bin'.format(epoch, args.trial), adj)
