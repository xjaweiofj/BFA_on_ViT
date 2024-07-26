import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from entropy.utils import *


import models
from models.quantization import quan_Conv2d, quan_Linear, quantize


class AttentionMap:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.feature = output

    def remove(self):
        self.hook.remove()

def differential_entropy(pdf, x_pdf):
    # pdf is a vector because we want to perform a numerical integration
    pdf = pdf + 1e-4
    f = -1 * pdf * torch.log(pdf)
    # Integrate using the composite trapezoidal rule
    ans = torch.trapz(f, x_pdf, dim=-1).mean()
    return ans


def get_image_prior_losses(inputs_jit):
    # Compute total variation regularization loss
    diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
    diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
    diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
    diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]

    loss_var_l2 = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
    return loss_var_l2


def clip(image_tensor, use_fp16=False):
    # Adjust the input based on mean and variance
    if use_fp16:
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float16)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float16)
    else:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
    for c in range(3):
        m, s = mean[c], std[c]
        image_tensor[:, c] = torch.clamp(image_tensor[:, c], -m / s, (1 - m) / s)
        #image_tensor[:, c] = torch.clamp(image_tensor[:, c], 0, 1)
    return image_tensor


def lr_policy(lr_fn):
    def _alr(optimizer, iteration, epoch):
        lr = lr_fn(iteration, epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    return _alr


def lr_cosine_policy(base_lr, warmup_length, epochs):
    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            e = epoch - warmup_length
            es = epochs - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        return lr

    return lr_policy(_lr_fn)



def entropy(batch_size, lr, model):

    # Load pretrained model
    p_model = build_model(model, Pretrained=True)

    # Hook the attention
    hooks = []
    if 'swin' in model:
        for m in p_model.layers:
            for n in range(len(m.blocks)):
                hooks.append(AttentionMap(m.blocks[n].attn.matmul2))
    else:
        for m in p_model.blocks:
            hooks.append(AttentionMap(m.attn.qkv))
            hooks.append(AttentionMap(m.attn.proj))

    # Init Gaussian noise
    img = torch.randn((batch_size, 3, 224, 224)).cuda()
    img.requires_grad = True

    # Init optimizer
    lr = 0.25 if 'swin' in model else 0.20
    optimizer = optim.Adam([img], lr=lr, betas=[0.5, 0.9], eps=1e-8)

    # Set pseudo labels
    pred = torch.LongTensor([random.randint(0, 999) for _ in range(batch_size)]).to('cuda')
    var_pred = random.uniform(2500, 3000)  # for batch_size 32

    criterion = nn.CrossEntropyLoss()

    # Train for two epochs
    for lr_it in range(2):
        if lr_it == 0:
            iterations_per_layer = 500
            lim = 15
        else:
            iterations_per_layer = 500
            lim = 30

        lr_scheduler = lr_cosine_policy(lr, 100, iterations_per_layer)

        with tqdm(range(iterations_per_layer)) as pbar:
            for itr in pbar:
                pbar.set_description(f"Epochs {lr_it+1}/{2}")

                # Learning rate scheduling
                lr_scheduler(optimizer, itr, itr)

                # Apply random jitter offsets (from DeepInversion[1])
                # [1] Yin, Hongxu, et al. "Dreaming to distill: Data-free knowledge transfer via deepinversion.", CVPR2020.
                off = random.randint(-lim, lim)
                img_jit = torch.roll(img, shifts=(off, off), dims=(2, 3))
                # Flipping
                flip = random.random() > 0.5
                if flip:
                    img_jit = torch.flip(img_jit, dims=(3,))

                # Forward pass
                optimizer.zero_grad()
                p_model.zero_grad()

                output = p_model(img_jit)

                loss_oh = criterion(output, pred)
                #loss_tv = torch.norm(get_image_prior_losses(img_jit) - var_pred)

                loss_entropy = 0
                for itr_hook in range(len(hooks)):
                    # Hook attention
                    # print (f"!!!!!!!!!!!!!!!!! hooks[itr_hook] = {hooks[itr_hook]}")
                    #print (f"!!!!!!!!!!!!!!!!! hooks = {hooks}")
                    attention = hooks[itr_hook].feature
                    # print (f"!!!!!!!!!!!!!!!!! attention = {attention}")
                    #print (f"type of attention = {type(attention)}")
                    #print (f"size of attention = {attention.shape}")
                    #print (f"number of dimension of attention = {attention.dim()}")
                    attention_p = attention[:, 1:, :]
                    #attention_p = attention.mean(dim=1)[:, 1:, :]
                    #attention_p = attention.mean(dim=1)[:, 1:]
                    sims = torch.cosine_similarity(attention_p.unsqueeze(1), attention_p.unsqueeze(2), dim=3)

                    # Compute differential entropy
                    kde = KernelDensityEstimator(sims.view(batch_size, -1))
                    start_p = sims.min().item()
                    end_p = sims.max().item()
                    x_plot = torch.linspace(start_p, end_p, steps=10).repeat(batch_size, 1).cuda()
                    kde_estimate = kde(x_plot)
                    dif_entropy_estimated = differential_entropy(kde_estimate, x_plot)
                    loss_entropy -= dif_entropy_estimated


    return loss_entropy


def main():

    model_list = ['deit_tiny_cifar10']
    #, 'deit_small_cifar10', 'deit_base_cifar10'] 
    lr = [1e-4, 1e-4, 1e-5]
    batch_size = 256
    
    i = 0
    for model in model_list:
        en = entropy(batch_size, lr[i], model)
        print (f"{model} entropy is {en}")

        i += 1


if __name__ == '__main__':
    main()
