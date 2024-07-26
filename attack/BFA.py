import torch
from models.quantization import quan_Conv2d, quan_Linear, quantize
import operator
from attack.data_conversion import *
import math
import torch.nn as nn

class BFA(object):
    def __init__(self, criterion, k_top=10):

        self.criterion = criterion
        # init a loss_dict to log the loss w.r.t each layer
        self.loss_dict = {}
        self.bit_counter = 0
        self.k_top = k_top
        self.n_bits2flip = 0
        self.loss = 0

    def flip_bit(self, m):
        '''
        the data type of input param is 32-bit floating, then return the data should
        be in the same data_type.
        '''

        # 1. flatten the gradient tensor to perform topk
        w_grad_topk, w_idx_topk = m.weight.grad.detach().abs().view(-1).topk(
            self.k_top) # flatten the weights into 1 single dimension, abstract them
        # update the b_grad to its signed representation
        w_grad_topk = m.weight.grad.detach().view(-1)[w_idx_topk] 
        # the w_grad_topk produced in step 1 is after abstraction, here use idx to get the original value
        
        max_value, max_index = torch.max(m.weight.detach().abs().view(-1), dim=0)
        min_value, min_index = torch.min(m.weight.detach().abs().view(-1), dim=0)


        w_topk = m.weight.detach().view(-1)[w_idx_topk] 
        # 2. create the b_grad matrix in shape of [N_bits, k_top]
        b_grad_topk = w_grad_topk * m.b_w.data

        # 3. generate the gradient mask to zero-out the bit-gradient
        # which can not be flipped
        b_grad_topk_sign = (b_grad_topk.sign() +
                            1) * 0.5  # zero -> negative, one -> positive
        
        # convert to twos complement into unsigned integer
        w_bin = int2bin(m.weight.detach().view(-1), m.N_bits).short() # w_bin is totally weight in 2's complement dtype
        w_bin_topk = w_bin[w_idx_topk]  # get the weights whose grads are topk
        
        # generate two's complement bit-map
        b_bin_topk = (w_bin_topk.repeat(m.N_bits,1) & m.b_w.abs().repeat(1,self.k_top).short()) \
        // m.b_w.abs().repeat(1,self.k_top).short()

        grad_mask = b_bin_topk ^ b_grad_topk_sign.short()
        # 4. apply the gradient mask upon ```b_grad_topk``` and in-place update it
        b_grad_topk *= grad_mask.float()
        # compare to the b_grad_topk in step 2, the elements whose location in grad_mask has value=0 are set to 0

        # 5. identify the several maximum of absolute bit gradient and return the
        # index, the number of bits to flip is self.n_bits2flip
        grad_max = b_grad_topk.abs().max()
        _, b_grad_max_idx = b_grad_topk.abs().view(-1).topk(self.n_bits2flip)

        # Flatten the tensor to make indexing easier
        flattened_tensor = b_grad_max_idx.flatten()

        # Generate a random index
        random_index = torch.randint(0, flattened_tensor.size(0), (1,))

        # Use the random index to select an element from the tensor
        selected_element = flattened_tensor[random_index]
        b_grad_max_idx = selected_element




        bit2flip = b_grad_topk.clone().view(-1).zero_() # copy the size of b_grad_topk, with all elements equal to 0
 
        if grad_max.item() != 0:  # ensure the max grad is not zero
            bit2flip[b_grad_max_idx] = 1
            bit2flip = bit2flip.view(b_grad_topk.size())
        else:
            bit2flip[b_grad_max_idx] = 1
            bit2flip = bit2flip.view(b_grad_topk.size())
            pass

        # find the gradient of the weight with chosen flipped bit
        chosen_idx = b_grad_max_idx.item() % self.k_top


        # 6. Based on the identified bit indexed by ```bit2flip```, generate another
        # mask, then perform the bitwise xor operation to realize the bit-flip.
        w_bin_topk_flipped = (bit2flip.short() * m.b_w.abs().short()).sum(0, dtype=torch.int16) \
                ^ w_bin_topk
        # w_bin_topk_flipped is w_bin_topk with the bits indicated by bit2flip is flipped
        # bit2flip is a n x k_top matric, n is the total bits in one weight (default n=8)
        # k_top represents the k_top weights selected, they have the higehst abs value, which means they are most vulnerable/important
        # if (i,j) element in bit2flip matric is 1, which means that the i_th bit in #j/k_top selected weights need to be flipped
        # if the #j weight value if 250d=11111010, i=0, then in w_bin_topk_flipped, this weight will become 122d=01111010


        # 7. update the weight in the original weight tensor
        w_bin[w_idx_topk] = w_bin_topk_flipped  # in-place change
        # update the flipped weight in w_bin (original weight in bin form)


        # convert bin back to int
        param_flipped = bin2int(w_bin,
                                m.N_bits).view(m.weight.data.size()).float()

        return param_flipped

    def progressive_bit_search(self, model, data, target):
        # PBS does one cross-layer search in PBS algorithm
        # model = NN network model
        # data = dataset features
        # target = dataset label
        ''' 
        Given the model, base on the current given data and target, go through
        all the layer and identify the bits to be flipped. 
        '''
        # Note that, attack has to be done in evaluation model due to batch-norm.
        # see: https://discuss.pytorch.org/t/what-does-model-eval-do-for-batchnorm-layer/7146
        model.eval()

        # 1. perform the inference w.r.t given data and target
        output = model(data)
        #         _, target = output.data.max(1)
        self.loss = self.criterion(output, target)

        # 2. zero out the grads first, then get the grads 
        for m in model.modules(): 
            # there is only weights in convolution layer and linear layer in DNN
            if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear): 
                if m.weight.grad is not None: 
                    m.weight.grad.data.zero_()

        self.loss.backward()
        # init the loss_max to enable the while loop
        self.loss_max = self.loss.item()
        
        ite_cnt = 0
        print (f"self.loss_max = {self.loss_max}")
        print (f"self.loss.item() = {self.loss.item()}")
        # 3. for each layer flip #bits = self.bits2flip (in-layer search)
        while self.loss_max <= self.loss.item(): # if loss in current layer is not the max loss, continue
            
            self.n_bits2flip += 1
            # iterate all the quantized conv and linear layer
            for name, module in model.named_modules(): # loop for every layer
                if isinstance(module, quan_Conv2d) or isinstance(module, quan_Linear): 

                    clean_weight = module.weight.data.detach() 
                    attack_weight = self.flip_bit(module) 
                    # change the weight to attacked weight and get loss
                    module.weight.data = attack_weight # update the flipped bits in weight
                    output = model(data)

                    self.loss_dict[name] = self.criterion(output,
                                                          target).item() # calculate and record the loss with flipped bit in current layer
                    # change the weight back to the clean weight
                    module.weight.data = clean_weight

            # after going through all the layer, now we find the layer with max loss
            # following lines: force the flipped bits fall in certain layers
            #filtered_loss_dict = {k: v for k, v in self.loss_dict.items() if 'attn' in k}
            #filtered_loss_dict = {k: v for k, v in self.loss_dict.items() if 'head' in k}
            #filtered_loss_dict = {k: v for k, v in self.loss_dict.items() if '0' not in k and '1' not in k and '2' not in k}
            #filtered_loss_dict = {k: v for k, v in self.loss_dict.items() if '9' not in k and '10' not in k and '11' not in k}


            max_loss_module = max(self.loss_dict.items(),
            #max_loss_module = max(filtered_loss_dict.items(),
                                  key=operator.itemgetter(1))[0]
            print (f"max_loss_module = {max_loss_module} in iteration {ite_cnt} of while loop")
            self.loss_max = self.loss_dict[max_loss_module] # update the loss_max
            
            ite_cnt += 1

            if (ite_cnt > 0):
                break
        # end of in-layer search
        
        # force it to flip head
        #max_loss_module = 'module.head'

        # 4. if the loss_max does lead to the degradation compared to the self.loss,
        # then change the that layer's weight without putting back the clean weight
        for name, module in model.named_modules():
            #if name == 'module.head':
            if name == max_loss_module:
                attack_weight = self.flip_bit(module)
                module.weight.data = attack_weight

        # reset the bits2flip back to 0
        self.bit_counter += self.n_bits2flip
        self.n_bits2flip = 0

        return
