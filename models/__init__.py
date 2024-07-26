# standard vit (load timm pretrained ckp for imagenet) 
from .standard_vit import vit_tiny, vit_small, vit_base 
from .standard_vit import deit_tiny, deit_small, deit_base 
from .standard_vit import swin_tiny, swin_small, swin_base

from .standard_vit import deit_base_10blocks_cifar100, deit_base_8blocks_cifar100, deit_base_6blocks_cifar100, deit_base_4blocks_cifar100, deit_base_2blocks_cifar100
from .standard_vit import deit_base_11blocks_cifar10, deit_base_10blocks_cifar10, deit_base_9blocks_cifar10, deit_base_8blocks_cifar10, deit_base_7blocks_cifar10, deit_base_6blocks_cifar10, deit_base_4blocks_cifar10, deit_base_2blocks_cifar10

from .standard_vit import vit_large, vit_huge, deit_huge

from .standard_vit import deit_base_10heads_cifar10_train


# standard vit (load the retrained ckp for cifar10)
from .load_ckp_vit import vit_tiny_cifar10, vit_small_cifar10, vit_base_cifar10
from .load_ckp_vit import deit_tiny_cifar10, deit_small_cifar10, deit_base_cifar10

from .load_ckp_vit import vit_tiny_cifar100, vit_small_cifar100, vit_base_cifar100
from .load_ckp_vit import deit_tiny_cifar100, deit_small_cifar100, deit_base_cifar100

from .load_ckp_vit import vit_tiny_imagenet100, vit_small_imagenet100, vit_base_imagenet100
from .load_ckp_vit import deit_tiny_imagenet100, deit_small_imagenet100, deit_base_imagenet100

from .load_ckp_vit import deit_base_10blk_cifar100, deit_base_8blk_cifar100, deit_base_6blk_cifar100, deit_base_4blk_cifar100, deit_base_2blk_cifar100
from .load_ckp_vit import deit_base_11blk_cifar10, deit_base_10blk_cifar10, deit_base_9blk_cifar10, deit_base_8blk_cifar10, deit_base_7blk_cifar10, deit_base_6blk_cifar10, deit_base_4blk_cifar10, deit_base_2blk_cifar10

from .load_ckp_vit import deit_tiny_cifar100_mlp, deit_small_cifar100_mlp, deit_base_cifar100_mlp
from .load_ckp_vit import deit_tiny_cifar100_attn, deit_small_cifar100_attn, deit_base_cifar100_attn
from .load_ckp_vit import deit_tiny_cifar100_patchembed, deit_small_cifar100_patchembed, deit_base_cifar100_patchembed
from .load_ckp_vit import deit_tiny_cifar100_head, deit_small_cifar100_head, deit_base_cifar100_head

