from __future__ import division
from __future__ import absolute_import
import pdb
import os, sys, shutil, time, random
import argparse
import torch
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset
import torchvision.transforms as transforms
from utils import AverageMeter, RecorderMeter, time_string, convert_secs2time
from tensorboardX import SummaryWriter
import models
from models.quantization import quan_Conv2d, quan_Linear, quantize

from attack.BFA import *
import torch.nn.functional as F
import copy

import torch.nn as nn
from functools import partial

from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import numpy as np

def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

################# Options ##################################################
############################################################################
parser = argparse.ArgumentParser(
    description='Training network for image classification',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--data_path', # dataset location
                    default='/data1/cifar-10-batches-py', # default='/home/elliot/data/pytorch/svhn/',
                    type=str,
                    help='Path to dataset')
parser.add_argument(
    '--dataset',
    type=str,
    choices=['cifar10', 'cifar100', 'imagenet', 'svhn', 'stl10', 'mnist', 'tiny_imagenet', 'imagenet-100'],
    help='Choose between Cifar10/100 and ImageNet.')
parser.add_argument('--arch',
                    metavar='ARCH',
                    default='lbcnn',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: resnext29_8_64)')
# Optimization options
parser.add_argument('--epochs',
                    type=int,
                    default=200,
                    help='Number of epochs to train.')
parser.add_argument('--model_name',
                    type=str,
                    default='None')
parser.add_argument('--sparsity',
                    type=str,
                    default='None')
parser.add_argument('--sparsity_th',
                    type=float,
                    default=0.0)
parser.add_argument('--optimizer',
                    type=str,
                    default='SGD',
                    choices=['SGD', 'Adam', 'AdamW' , 'YF'])
parser.add_argument('--test_batch_size',
                    type=int,
                    default=256,
                    help='Batch size.')
parser.add_argument('--learning_rate',
                    type=float,
                    default=0.001,
                    help='The Learning Rate.')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
'''
parser.add_argument('--num_heads',
                    type=int,
                    default=12,
                    help='number of heads.')
'''
parser.add_argument('--decay',
                    type=float,
                    default=1e-4,
                    help='Weight decay (L2 penalty).')
parser.add_argument('--schedule',
                    type=int,
                    nargs='+',
                    default=[80, 120],
                    help='Decrease learning rate at these epochs.')
parser.add_argument(
    '--gammas',
    type=float,
    nargs='+',
    default=[0.1, 0.1],
    help=
    'LR is multiplied by gamma on schedule, number of gammas should be equal to schedule'
)
# Checkpoints
parser.add_argument('--print_freq',
                    default=100,
                    type=int,
                    metavar='N',
                    help='print frequency (default: 200)')
parser.add_argument('--save_path',
                    type=str,
                    default='/data1/Xuan_vit_ckp/',
                    help='Folder to save checkpoints and log.')
parser.add_argument('--resume',
                    default='',
                    type=str,
                    metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--start_epoch',
                    default=0,
                    type=int,
                    metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--evaluate',
                    dest='evaluate',
                    action='store_true',
                    help='evaluate model on validation set')
parser.add_argument(
    '--fine_tune',
    dest='fine_tune',
    action='store_true',
    help='fine tuning from the pre-trained model, force the start epoch be zero'
)
parser.add_argument('--model_only',
                    dest='model_only',
                    action='store_true',
                    help='only save the model without external utils_')
# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--gpu_id',
                    type=int,
                    default=0,
                    help='device range [0,ngpu-1]')
parser.add_argument('--workers',
                    type=int,
                    default=4,
                    help='number of data loading workers (default: 2)')
# random seed
parser.add_argument('--manualSeed', type=int, default=None, help='manual seed')
# quantization
parser.add_argument(
    '--reset_weight',
    dest='reset_weight',
    action='store_true',
    help='enable the weight replacement with the quantized weight')
parser.add_argument(
    '--optimize_step',
    dest='optimize_step',
    action='store_true',
    help='enable the step size optimization for weight quantization')
# Bit Flip Attacked
parser.add_argument('--bfa',
                    dest='enable_bfa',
                    action='store_true',
                    help='enable the bit-flip attack')
parser.add_argument('--attack_sample_size',
                    type=int,
                    default=128,
                    help='attack sample size')
parser.add_argument('--n_iter',
                    type=int,
                    default=20,
                    help='number of attack iterations')
parser.add_argument(
    '--k_top',
    type=int,
    default=10,
    help='k weight with top ranking gradient used for bit-level gradient check.'
) # only take the bits in the k weight with top ranking gradient for BFA?

##########################################################################

args = parser.parse_args()

# added by Xuan to fix the seed
setup_seed(seed=args.manualSeed)


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
if args.ngpu == 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(
        args.gpu_id)  # make only device #gpu_id visible, then

args.use_cuda = args.ngpu > 0 and torch.cuda.is_available()  # check GPU

# Give a random seed if no manual configuration
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)

if args.use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

cudnn.benchmark = True

###############################################################################
###############################################################################
# def main

def main():
    # Init logger6
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    log = open(
        os.path.join(args.save_path,
                     'log_seed_{}.txt'.format(args.manualSeed)), 'w')
    print_log('save path : {}'.format(args.save_path), log)
    state = {k: v for k, v in args._get_kwargs()}
    print_log(state, log)
    print_log("Random Seed: {}".format(args.manualSeed), log)
    print_log("python version : {}".format(sys.version.replace('\n', ' ')),
              log)
    print_log("torch  version : {}".format(torch.__version__), log)
    print_log("cudnn  version : {}".format(torch.backends.cudnn.version()),
              log)

    # Init the tensorboard path and writer
    tb_path = os.path.join(args.save_path, 'tb_log',
                           'run_' + str(args.manualSeed))
    # logger = Logger(tb_path)
    writer = SummaryWriter(tb_path)

    # Init dataset
    if not os.path.isdir(args.data_path):
        os.makedirs(args.data_path)
    
    
    # settings for each dataset
    if args.dataset == 'cifar10':
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
    elif args.dataset == 'cifar100':
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std = [x / 255 for x in [68.2, 65.4, 70.4]]
    elif args.dataset == 'svhn':
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    elif args.dataset == 'mnist':
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    elif args.dataset == 'imagenet' or args.dataset == 'tiny_imagenet' or args.dataset == 'imagenet-100':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        assert False, "Unknow dataset : {}".format(args.dataset)

    if args.dataset == 'imagenet' or args.dataset == 'tiny_imagenet' or args.dataset == 'imagenet-100':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])  # here is actually the validation dataset
    else:
        train_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(), # convert image input to tensor
            transforms.Normalize(mean, std) # normalize the weights
        ])
        train_transform=build_transform(False, 224) # this will transform the size of image to 224x224, if we use cifar10, it will still output image with size 224x224

        test_transform = transforms.Compose(
            [transforms.Resize(224),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
             # transforms.Normalize(mean, std)
             ])

        test_transform=build_transform(False, 224)

    if args.dataset == 'mnist':
        train_data = dset.MNIST(args.data_path,
                                train=True,
                                transform=train_transform,
                                download=True)
        test_data = dset.MNIST(args.data_path,
                               train=False,
                               transform=test_transform,
                               download=True)
        num_classes = 10
    elif args.dataset == 'cifar10':
        train_data = dset.CIFAR10(args.data_path,
                                  train=True,
                                  transform=train_transform,
                                  download=True)
        test_data = dset.CIFAR10(args.data_path,
                                 train=False,
                                 transform=test_transform,
                                 download=True)
        num_classes = 10
        img, label = train_data[0]

    elif args.dataset == 'cifar100':
        train_data = dset.CIFAR100(args.data_path,
                                   train=True,
                                   transform=train_transform,
                                   download=True)
        test_data = dset.CIFAR100(args.data_path,
                                  train=False,
                                  transform=test_transform,
                                  download=True)
        num_classes = 100
    elif args.dataset == 'svhn':
        train_data = dset.SVHN(args.data_path,
                               split='train',
                               transform=train_transform,
                               download=True)
        test_data = dset.SVHN(args.data_path,
                              split='test',
                              transform=test_transform,
                              download=True)
        num_classes = 10
    elif args.dataset == 'stl10':
        train_data = dset.STL10(args.data_path,
                                split='train',
                                transform=train_transform,
                                download=True)
        test_data = dset.STL10(args.data_path,
                               split='test',
                               transform=test_transform,
                               download=True)
        num_classes = 10
    elif args.dataset == 'imagenet':
        train_dir = os.path.join(args.data_path, 'train')
        test_dir = os.path.join(args.data_path, 'val')
        train_data = dset.ImageFolder(train_dir, transform=train_transform)
        test_data = dset.ImageFolder(test_dir, transform=test_transform)
        num_classes = 1000
    elif args.dataset == 'tiny_imagenet':
        train_dir = os.path.join(args.data_path, 'train')
        test_dir = os.path.join(args.data_path, 'val')
        train_data = dset.ImageFolder(train_dir, transform=train_transform)
        test_data = dset.ImageFolder(test_dir, transform=test_transform)
        num_classes = 200
    elif args.dataset == 'imagenet-100':
        train_dir = os.path.join(args.data_path, 'train')
        test_dir = os.path.join(args.data_path, 'val')
        train_data = dset.ImageFolder(train_dir, transform=train_transform)
        test_data = dset.ImageFolder(test_dir, transform=test_transform)
        num_classes = 100
    else:
        assert False, 'Do not support dataset : {}'.format(args.dataset)

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.attack_sample_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True)


    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=args.test_batch_size,
                                              shuffle=True,
                                              num_workers=args.workers,
                                              pin_memory=True)

    print_log("=> creating model '{}'".format(args.arch), log)

    # Init model, criterion, and optimizer

    net = models.__dict__[args.arch](num_classes) # choose a NN model, resnet34_quan in default, can be modified in .sh file

    print_log("=> network :\n {}".format(net), log)
    
    if args.use_cuda:
        if args.ngpu > 1:
            net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss() # define a common cross entropy loss as loss function

    
    # separate the parameters thus param groups can be updated by different optimizer
    all_param = [
        param for name, param in net.named_parameters()
        if not 'step_size' in name
    ]

    step_param = [
        param for name, param in net.named_parameters() if 'step_size' in name
    ]

    if args.optimizer == "SGD":
        print("using SGD as optimizer")
        optimizer = torch.optim.SGD(all_param,
                                    lr=state['learning_rate'],
                                    momentum=state['momentum'],
                                    weight_decay=state['decay'],
                                    nesterov=True)

    elif args.optimizer == "Adam":
        print("using Adam as optimizer")
        optimizer = torch.optim.Adam(filter(lambda param: param.requires_grad,
                                            net.parameters()),
                                     lr=state['learning_rate'],
                                     weight_decay=state['decay'])

    elif args.optimizer == "AdamW":
        print(f"using AdamW as optimizer, lr = {state['learning_rate']}")
        optimizer = torch.optim.AdamW(filter(lambda param: param.requires_grad,
                                            net.parameters()),
                                     lr=state['learning_rate'],
                                     weight_decay=state['decay'])

    elif args.optimizer == "RMSprop":
        print("using RMSprop as optimizer")
        optimizer = torch.optim.RMSprop(
            filter(lambda param: param.requires_grad, net.parameters()),
            lr=state['learning_rate'],
            alpha=0.99,
            eps=1e-08,
            weight_decay=0,
            momentum=0)

    if args.use_cuda:
        net.cuda()
        criterion.cuda()

    recorder = RecorderMeter(args.epochs)  # count number of epoches

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print_log("=> loading checkpoint '{}'".format(args.resume), log)
            checkpoint = torch.load(args.resume)
            if not (args.fine_tune):
                args.start_epoch = checkpoint['epoch']
                recorder = checkpoint['recorder']
                optimizer.load_state_dict(checkpoint['optimizer'])

            state_tmp = net.state_dict()
            if 'state_dict' in checkpoint.keys():
                state_tmp.update(checkpoint['state_dict'])
            else:
                state_tmp.update(checkpoint)

            net.load_state_dict(state_tmp)

            print_log(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, args.start_epoch), log)
        else:
            print_log("=> no checkpoint found at '{}'".format(args.resume),
                      log)
    else:
        print_log(
            "=> do not use any checkpoint for {} model".format(args.arch), log)

    # update the step_size once the model is loaded. This is used for quantization.
    for m in net.modules(): 
        if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear):
            # simple step size update based on the pretrained model or weight init
            m.__reset_stepsize__()

    # block for quantizer optimization
    if args.optimize_step:
        print (f"step_param = {step_param}") 
        optimizer_quan = torch.optim.SGD(step_param,
                                         lr=0.01,
                                         momentum=0.9,
                                         weight_decay=0,
                                         nesterov=True)

        for m in net.modules():
            if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                for i in range(
                        300
                ):  # runs 200 iterations to reduce quantization error
                    optimizer_quan.zero_grad()
                    weight_quan = quantize(m.weight, m.step_size,
                                    m.half_lvls) * m.step_size
                    loss_quan = F.mse_loss(weight_quan,
                                           m.weight,
                                           reduction='mean')
                    loss_quan.backward()
                    optimizer_quan.step()

        for m in net.modules():
            if isinstance(m, quan_Conv2d) or isinstance(m, nn.Conv2d):
                print(m.step_size.data.item(),
                      (m.step_size.detach() * m.half_lvls).item(),
                      m.weight.max().item())

    
    
    # Main loop
    start_time = time.time()
    epoch_time = AverageMeter()
    
    print (f"There are {args.epochs} epochs in total in the loop in the main loop, will start with epoch={args.start_epoch}")
    for epoch in range(args.start_epoch, args.epochs):
        current_learning_rate, current_momentum = adjust_learning_rate(
            optimizer, epoch, args.gammas, args.schedule)
        # Display simulation time
        need_hour, need_mins, need_secs = convert_secs2time(
            epoch_time.avg * (args.epochs - epoch))
        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(
            need_hour, need_mins, need_secs)

        print (log)
        print ("The log file is -- " + str(log)) 
        
        print (f"\targs.epochs={args.epochs}")
        print ("\targs.epochs={:03d}".format(args.epochs))
        print_log(
            '\n==>>{:s} [Epoch={:03d}/{:03d}] {:s} [LR={:6.4f}][M={:1.2f}]'.format(time_string(), epoch, args.epochs,
                                                                                   need_time, current_learning_rate,
                                                                                   current_momentum) \
            + ' [Best : Accuracy={:.2f}, Error={:.2f}]'.format(recorder.max_accuracy(False),
                                                               100 - recorder.max_accuracy(False)), log)

        # train for one epoch
        print (f"epoch={epoch}, dive ointo train function in next line")
        train_acc, train_los = train(train_loader, net, criterion, optimizer, # net is the chosen NN for experiments
                                     epoch, log)

        # evaluate on validation set
        val_acc, _, val_los = validate(test_loader, net, criterion, log)
        recorder.update(epoch, train_los, train_acc, val_los, val_acc)
        is_best = val_acc >= recorder.max_accuracy(False)

        if args.model_only:
            checkpoint_state = {'state_dict': net.state_dict}
        else:
            checkpoint_state = {
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': net.state_dict(),
                'recorder': recorder,
                'optimizer': optimizer.state_dict(),
            }

        save_checkpoint(checkpoint_state, is_best, args.save_path,
                        'checkpoint.pth.tar', log)

        # measure elapsed time
        epoch_time.update(time.time() - start_time)
        start_time = time.time()
        recorder.plot_curve(os.path.join(args.save_path, 'curve.png'))

        # save addition accuracy log for plotting
        accuracy_logger(base_dir=args.save_path,
                        epoch=epoch,
                        train_accuracy=train_acc,
                        test_accuracy=val_acc)
        
    print (f"========================= Evaluate the model before reset weight =======================")
    if args.evaluate:  
        validate(test_loader, net, criterion, log)
    
    # block for weight reset
    if args.reset_weight:
        for m in net.modules():
            if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear):
                m.__reset_weight__()
                
    
    
    # start BFA here, the definition of class BFA can be found in BFA.py
    attacker = BFA(criterion, args.k_top) 
    net_clean = copy.deepcopy(net)
    
    if args.enable_bfa:
        # perform_attack can be found in main.py
        perform_attack(attacker, net, net_clean, train_loader, test_loader,
                    args.n_iter, log, writer) 
    
    # only with --evaluate will this part be triggered for evaluating for validation set 
    print (f"========================= Evaluate the model after attack =======================")
    if args.evaluate:  
        validate(test_loader, net, criterion, log)
        #return

    
    log.close()


def perform_attack(attacker, model, model_clean, train_loader, test_loader,
        N_iter, log, writer): # N_iter is n_iter defined in .sh file
    # perform_attack is the whole Bit-Flip attack which contains # N_iter iterations of cross-layer search
    # Note that, attack has to be done in evaluation model due to batch-norm.
    # see: https://discuss.pytorch.org/t/what-does-model-eval-do-for-batchnorm-layer/7146
    model.eval()
    losses = AverageMeter() # AverageMeter is an object, its definition can be found in utils.py, convenient for sum, count, avg calculation
    iter_time = AverageMeter()
    attack_time = AverageMeter()

    # attempt to use the training data to conduct BFA
    for _, (data, target) in enumerate(train_loader):
        if args.use_cuda:
            target = target.cuda(non_blocking=True)
            data = data.cuda()

        # Override the target to prevent label leaking
        # print (f"data type = {type(data)}, data size = {data.shape}")
        _, target = model(data).data.max(1)
        break

    # evaluate the test accuracy of clean model
    val_acc_top1, val_acc_top5, val_loss = validate(test_loader, model,
                                                    attacker.criterion, log)

    writer.add_scalar('attack/val_top1_acc', val_acc_top1, 0)
    writer.add_scalar('attack/val_top5_acc', val_acc_top5, 0)
    writer.add_scalar('attack/val_loss', val_loss, 0)

    print_log('k_top is set to {}'.format(args.k_top), log)
    print_log('Attack sample size is {}'.format(data.size()[0]), log)
    end = time.time()
    
    # perform # N_iter iterations of cross-layer search 
    for i_iter in range(N_iter):
        print_log('**********************************', log)

        attacker.progressive_bit_search(model, data, target) # PBS algorithm, which contains 1 iteration of cross-layer (outer loop)

        # measure data loading time
        attack_time.update(time.time() - end)
        end = time.time()

        h_dist = hamming_distance(model, model_clean)

        # record the loss 
        losses.update(attacker.loss_max, data.size(0)) 
                                        # size(0) = 1
        
        print_log(
            'Iteration: [{:03d}/{:03d}]   '
            'Attack Time {attack_time.val:.3f} ({attack_time.avg:.3f})  '.
            format((i_iter + 1),
                   N_iter,
                   attack_time=attack_time,
                   iter_time=iter_time) + time_string(), log)

        print_log('loss before attack: {:.4f}'.format(attacker.loss.item()),
                  log)
        print_log('loss after attack: {:.4f}'.format(attacker.loss_max), log)
        print_log('bit flips: {:.0f}'.format(attacker.bit_counter), log)
        print_log('hamming_dist: {:.0f}'.format(h_dist), log)

        writer.add_scalar('attack/bit_flip', attacker.bit_counter, i_iter + 1)
        writer.add_scalar('attack/h_dist', h_dist, i_iter + 1)
        writer.add_scalar('attack/sample_loss', losses.avg, i_iter + 1)

        # exam the BFA on entire val dataset
        # test the accuracy and loss after flip the bit in this iteration
        val_acc_top1, val_acc_top5, val_loss = validate(
            test_loader, model, attacker.criterion, log)

        writer.add_scalar('attack/val_top1_acc', val_acc_top1, i_iter + 1)
        writer.add_scalar('attack/val_top5_acc', val_acc_top5, i_iter + 1)
        writer.add_scalar('attack/val_loss', val_loss, i_iter + 1)

        # measure elapsed time
        iter_time.update(time.time() - end)
        print_log(
            'iteration Time {iter_time.val:.3f} ({iter_time.avg:.3f})'.format(
                iter_time=iter_time), log)
        end = time.time()

    return


# train function (forward, backward, update)
def train(train_loader, model, criterion, optimizer, epoch, log):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.use_cuda:
            target = target.cuda(
                non_blocking=True
            )  # the copy will be asynchronous with respect to the host.
            input = input.cuda()

        # compute output
        output = model(input)  # this line is used to calculate the output of NN
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print_log(
                '  Epoch: [{:03d}][{:03d}/{:03d}]   '
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})   '
                'Data {data_time.val:.3f} ({data_time.avg:.3f})   '
                'Loss {loss.val:.4f} ({loss.avg:.4f})   '
                'Prec@1 {top1.val:.3f} ({top1.avg:.3f})   '
                'Prec@5 {top5.val:.3f} ({top5.avg:.3f})   '.format(
                    epoch,
                    i,
                    len(train_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    top1=top1,
                    top5=top5) + time_string(), log)
    print_log(
        '  **Train** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'
        .format(top1=top1, top5=top5, error1=100 - top1.avg), log)
    return top1.avg, losses.avg


# this fucntion is used to calcualte the sparsity
def hook_fn(module, input, output):
    sparsity = (abs(output) <= args.sparsity_th).float().mean().item()
    print(f"GELU Sparsity: {sparsity * 100:.2f}%")

def calculate_final_output_sparsity(output):
    sparsity = (abs(output) <= args.sparsity_th).float().mean().item()
    print(f"Final Output Sparsity: {sparsity * 100:.2f}%")

def validate(val_loader, model, criterion, log):

    # Adjust for DataParallel wrapper
    model_to_use = model.module if hasattr(model, 'module') else model

    # Register hooks for GELU activation layers
    handles = []

    if args.sparsity == 'True':
        # Assuming each 'Block' in 'blocks' contains a GELU activation in its 'mlp' component
        
        # not sure why but there is error here
        # this is error-free
        for module_idx, block in enumerate(model_to_use.blocks):
            handle = model_to_use.blocks[module_idx].mlp.act.register_forward_hook(hook_fn) # sparsity handle for GELU output
            # handle = model_to_use.blocks[module_idx].attn.qkv.register_forward_hook(hook_fn) # sparsity handle for qkv output
            handles.append(handle)


        # print (handles)

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    top10 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            if args.use_cuda:
                target = target.cuda(non_blocking=True)
                input = input.cuda()

            # compute output
            output = model(input) 

            if args.sparsity == 'True':
                # Calculate sparsity of the final output
                calculate_final_output_sparsity(output)

            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5, prec10 = accuracy(output.data, target, topk=(1, 5, 10))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))
            top10.update(prec10.item(), input.size(0))

        print_log(
            '  **Test** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Prec@10 {top10.avg:.3f} Error@1 {error1:.3f}'
            .format(top1=top1, top5=top5, top10=top10, error1=100 - top1.avg), log)

    if args.sparsity == 'True':
        # Remove hooks after validation to avoid affecting other operations
        for handle in handles:
            handle.remove()

    # to avoid more changes, top10 is only printed to log file, but not returned 
    return top1.avg, top5.avg, losses.avg


def print_log(print_string, log):
    print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()


def save_checkpoint(state, is_best, save_path, filename, log):
    filename = os.path.join(save_path, filename)
    torch.save(state, filename)
    if is_best:  # copy the checkpoint to the best model if it is the best_accuracy
        bestname = os.path.join(save_path, 'model_best.pth.tar')
        shutil.copyfile(filename, bestname)
        print_log("=> Obtain best accuracy, and update the best model", log)


def adjust_learning_rate(optimizer, epoch, gammas, schedule):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.learning_rate
    mu = args.momentum

    if args.optimizer != "YF":
        assert len(gammas) == len(
            schedule), "length of gammas and schedule should be equal"
        for (gamma, step) in zip(gammas, schedule):
            if (epoch >= step):
                lr = lr * gamma
            else:
                break
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    elif args.optimizer == "YF":
        lr = optimizer._lr
        mu = optimizer._mu

    return lr, mu


def accuracy(output, target, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        # print (f"maxk={maxk}")
        
        batch_size = target.size(0)
        # print (f"batch_size={batch_size}")

        # _ is discard (do not give a var name)
        # 2nd output of topk, is the index list of the topk elements
        # 0 always infer column, 1 row
        # maxk is the topk elements that we want, 1 infer
        # 1st True means max insatead of min, 2nd True means the returned k elements are themselves sorted
        _, pred = output.topk(maxk, 1, True, True)
        # output should be the smp results, topk picks the top k possibilities for each input token
        # print (f"output={output}") # size = # samples * # classes 
        # print (f"pred={pred}") # size = # samples * # topk (stores the indexes of top k classes with highest smp for each sample) 
        pred = pred.t() # size = # topk * # samples
        # print (f"pred.t={pred.t}")
        # now each column of the pred shows the top k classes of prediction, any of them if matched, then topk can be considered as correct. But only the first row in a column matches, then top1 matches

        # eq(): compare 2 tensor of the same size, True if same, False if not
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        # target size = 1 * # samples
        # each column can have at most 1 True (one sample cannot have several different predictions that are all correct)
        # view(x,y): view the given numpy array in x rows, when y=-1 it is a placeholder, which means the # column is decided automatically

        # print (f"target={target}")
        # target size = 1 * # samples, labels of all samples
        # print (f"target.view(1,-1).expand_as(pred)={target.view(1,-1).expand_as(pred)}")
        # duplicate the 1-row target matrix into # topk rows (each row is the same)
        # print (f"correct={correct}")
        # size = # topk * # samples

        res = []
        for k in topk:
            '''
            print ("Here print correct[:k] !!!!!!!!!!!!!!!!!!!")
            print (correct[:k])
            print (f"k={k}")
            print (correct[:k].size())
            print (type(correct[:k]))
            '''
            # print (f"**************************** k={k} begins ********************************")
            correct_k = correct[:k].reshape(-1).float().sum(0) # [:k] means keep the array with rows before k, no cutting for other dim
            # sum up the matrix correct_k to calculate the # of total matches

            # print (f"correct[:{k}]={correct[:k]}")
            # print (f"correct[:{k}].reshape(-1)={correct[:k].reshape(-1)}")
            # print (f"correct_k=correct[:{k}].reshape(-1).float().sum(0)={correct_k}")
            res.append(correct_k.mul_(100.0 / batch_size))
            # recall the accuracy that we match with the accuracy after training, the accuracy should be the common accuracy definition
            # print (f"res={res}")
            # print (f"**************************** k={k} ends ********************************")
        return res


def accuracy_logger(base_dir, epoch, train_accuracy, test_accuracy):
    file_name = 'accuracy.txt'
    file_path = "%s/%s" % (base_dir, file_name)
    # create and format the log file if it does not exists
    if not os.path.exists(file_path):
        create_log = open(file_path, 'w')
        create_log.write('epochs train test\n')
        create_log.close()

    recorder = {}
    recorder['epoch'] = epoch
    recorder['train'] = train_accuracy
    recorder['test'] = test_accuracy
    # append the epoch index, train accuracy and test accuracy:
    with open(file_path, 'a') as accuracy_log:
        accuracy_log.write(
            '{epoch}       {train}    {test}\n'.format(**recorder))

def build_transform(is_train, input_size):
    resize_im = input_size > 32

    t = []
    if resize_im:
        size = int(input_size / 0.875)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)


if __name__ == '__main__':
    main()
