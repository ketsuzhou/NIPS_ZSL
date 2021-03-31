import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.flow.act_norm import ActNorm
from models.flow.inv_conv import InvConv
from models.flow.nn import GatedConv
from models.flow.flowpp_coupling import Flowpp_Coupling
from models.flow.dct_transform import DCTPooling2d
from models.flow.reshapes import *
from util import channelwise, checkerboard, Flip, safe_log, squeeze, unsqueeze
import FrEIA.framework as Ff
import FrEIA.modules as Fm
from FrEIA.framework import ConditionNode

clamping = 1.5
feature_channels = 256
fc_cond_length = 512
# n_blocks_fc = 8
img_dims_orig = (256, 256)
img_dims = (img_dims_orig[0] // 4, img_dims_orig[0] // 4)

def cond_subnet(level, c_out, extra_conv=False):
    c_intern = [feature_channels, 128, 128, 256]
    modules = []

    for i in range(level):
        modules.extend([nn.Conv2d(c_intern[i], c_intern[i+1], 3, stride=2, padding=1),
                        nn.LeakyReLU() ])
    if extra_conv:
        modules.extend([
            nn.Conv2d(c_intern[level], 128, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(128, 2*c_out, 3, padding=1),
        ])
    else:
        modules.append(nn.Conv2d(c_intern[level], 2*c_out, 3, padding=1))

    modules.append(nn.BatchNorm2d(2*c_out))

    return nn.Sequential(*modules)

class invertible_net(Ff.ReversibleGraphNet):
    def __init__(self, condition, use_attn, use_split,  downsample, FlowBlocks_architecture,
                in_shape, in_shape_condition_node, verbose=True, mid_channels=96, num_ConvAttnBlock=10,
                num_components=32, drop_prob=0.2, num_InvAutoFC=1):
        self.in_shape = in_shape
        nodes = [Ff.InputNode(*in_shape, name='Input')]
        for block_index in range(len(FlowBlocks_architecture)):
            if in_shape_condition_node is not None:
                ConditionNode = Ff.ConditionNode(*in_shape_condition_node, name=f'Condition_node_{block_index}') if in_shape_condition_node
            else:
                

            nodes = Flow_Block(nodes, ConditionNode, block_index,
                                in_shape=in_shape,
                                FlowBlocks_architecture=FlowBlocks_architecture[block_index],
                                mid_channels=mid_channels,
                                num_ConvAttnBlock=num_ConvAttnBlock,
                                num_components=num_components,
                                use_attn=use_attn[block_index],
                                drop_prob=drop_prob)
            nodes, in_shape = reduce_spacial_dimension(nodes, 
                                in_shape, block_index, 
                                use_split=use_split[block_index], 
                                downsample=downsample[block_index])

        for _ in range(num_InvAutoFC):
            nodes.append(Ff.Node(nodes[-1].out0, Fm.ActNorm, 
                                {}, name=f'ActNorm_final'))
            nodes.append(Ff.Node(nodes[-1].out0, Fm.InvAutoFC, 
                                {}, name='InvAutoFC'))

        nodes.append(Ff.OutputNode(nodes[-1].out0, name='output'))
        # print([i.name for i in nodes])

        super().__init__(nodes, verbose=verbose)
        self.invertible_net = Ff.ReversibleGraphNet()


def Flow_Block(nodes, ConditionNode, block_index, in_shape, FlowBlocks_architecture, mid_channels, num_ConvAttnBlock, num_components, drop_prob, use_self_attn):
        num_channelwise, num_checkerboard = FlowBlocks_architecture
        
        for i in range(num_channelwise):
            nodes.append(Ff.Node(nodes[-1].out0, Fm.ActNorm, {}, 
                                name=f'ActNorm_{block_index}_{i}'))
            nodes.append(Ff.Node([nodes[-1].out0], Fm.PermuteRandom, 
                                {'seed':np.random.randint(2**31)},
                                name=f'PermuteRandom_{block_index}_{i}'))
            if use_self_attn==1:
                nodes.append(Ff.Node(nodes[-1].out0, Flowpp_Coupling, {
                                'mid_channels': mid_channels,
                                'num_ConvAttnBlock': num_ConvAttnBlock,
                                'num_components': num_components,
                                'use_attn': use_self_attn,
                                'drop_prob': drop_prob}, 
                                conditions=ConditionNode, 
                                name=f'FlowppCoupling_{block_index}_{i}')
                                )
            else:
                nodes.append(Ff.Node([nodes[-1].out0], Fm.AffineCouplingOneSided,{
                                'dims_in': dims_in,
                                'dims_c': dims_c,
                                'subnet_constructor': Fm.F_conv(in_channels, channels, 
                                    channels_hidden=None, stride=None, kernel_size=3, leaky_slope=0.1, batch_norm=False), 
                                'clamp': 5.},
                                conditions=ConditionNode, 
                                name=F'conv_{block_index}_{i}')
                                ) 
                            
        # for i in range(num_checkerboard):
        #     nodes.append(Ff.Node(nodes[-1].out0, Fm.Reshape, {}, 
        #                         name=f'reshape_{block_index}_{i}'))
        #     nodes.append(Ff.Node(nodes[-1].out0, Fm.ActNorm, {}, 
        #                         name=f'ActNorm_{block_index}_{i}'))
        #     nodes.append(Ff.Node([nodes[-1].out0], Fm.PermuteRandom, 
        #                         {'seed':np.random.randint(2**31)},
        #                         name=f'PermuteRandom_{block_index}_{i}'))
        #     if use_self_attn==1:
        #         nodes.append(Ff.Node(nodes[-1].out0, Flowpp_Coupling, {
        #                         'mid_channels': mid_channels,
        #                         'num_ConvAttnBlock': num_ConvAttnBlock,
        #                         'num_components': num_components,
        #                         'use_attn': use_self_attn,
        #                         'drop_prob': drop_prob}, 
        #                         conditions=ConditionNode, 
        #                         name=F'FlowppCoupling_{block_index}_{i}'))
        #     else:
        #         nodes.append(Ff.Node([nodes[-1].out0], Fm.AffineCouplingOneSided,{
        #                         'dims_in': dims_in,
        #                         'dims_c': dims_c,
        #                         'subnet_constructor': Fm.F_conv(in_channels, channels, 
        #                             channels_hidden=None, stride=None, kernel_size=3, leaky_slope=0.1, batch_norm=False), 
        #                         'clamp': 5.},
        #                         conditions=ConditionNode, 
        #                         name=F'conv_{block_index}_{i}')
        #                         )                          
        #     nodes.append(Ff.Node(nodes[-1].out0, Fm.Reshape, {}, 
        #                         name=f'reshape_{block_index}_{i}'))
        return nodes

def get_Downsampled_shape(in_shape):
    in_channels, weight, height = in_shape
    in_shape = in_channels * 4, weight // 2, height // 2
    return in_shape

def get_Poolinged_shape(in_shape):
    in_channels, weight, height = in_shape
    in_shape = in_channels * weight * height, 1, 1
    return in_shape
        
def get_Splited_shape(in_shape):
    in_channels, weight, height = in_shape
    in_shape = in_channels //2 , weight, height
    return in_shape

def reduce_spacial_dimension(nodes, in_shape, block_index , use_split, downsample=None):
    # Downsampling
    if downsample=='Haar':
        nodes.append(Ff.Node([nodes[-1].out0], Fm.HaarDownsampling, 
                        {'rebalance':0.5, 'order_by_wavelet':True}, 
                        name='Haar'))
        in_shape = get_Poolinged_shape(in_shape)
    if downsample=='Checkboard':
        nodes.append(Ff.Node([nodes[-1].out0], Fm.IRevNetDownsampling, {}, 
                        name='Checkboard'))
        in_shape = get_Poolinged_shape(in_shape)
    # Global pooling
    if downsample=='Flatten':
        nodes.append(Ff.Node(nodes[-1].out0, Fm.Flatten, {}, 
                        name='Flatten'))
        in_shape = get_Downsampled_shape(in_shape)
    if downsample=='DCTPooling':
        nodes.append(Ff.Node([nodes[-1].out0], DCTPooling2d, {'rebalance':0.5}, 
                        name='DCT_Pooling'))
        in_shape = get_Downsampled_shape(in_shape)
    # Do not reduce the spacial dimensions
    if downsample=='None':
        pass

    if use_split:
        # reduce the channel dimensions
        in_shape = get_Splited_shape(in_shape)
        Split = Ff.Node([nodes[-1].out0], Fm.Split1D,
                    {'split_size_or_sections': (in_shape[0], in_shape[0]), 'dim':0}, 
                    name=f'split_{block_index}')
        nodes.append(Ff.Node([Split.out1], Fm.Flatten, {}, name='flatten'))
        nodes.append(Ff.OutputNode([nodes[-1].out0], name='out'))
        nodes.append(Split)
        
        # nodes.append(Ff.Node([nodes[-1].out0], Fm.Split1D,
        #                 {'split_size_or_sections': use_split, 'dim':0}, name='split'))
        # output = Ff.Node([nodes[-1].out1], flattening_layer, {}, name='flatten')
        # nodes.insert(-2, output)
        # nodes.insert(-2, Ff.OutputNode([output.out0], name='out'))
    return nodes, in_shape

