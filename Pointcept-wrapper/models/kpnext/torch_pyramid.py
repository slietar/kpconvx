#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
# ----------------------------------------------------------------------------------------------------------------------
#
#   Hugues THOMAS - 06/10/2023
#
#   KPConvX project: torch_pyramid.py
#       > input pyramid strucure of subsamplings and neighbors
#


import torch
from torch import Tensor
from typing import Tuple, List
from easydict import EasyDict

from utils.gpu_subsampling import subsample_pack_batch

from utils.gpu_neigbors import radius_search_pack_mode, keops_radius_count
from utils.cpp_funcs import batch_knn_neighbors, batch_grid_partition



# ----------------------------------------------------------------------------------------------------------------------
#
#           Input pyramid functions
#       \*****************************/
#




@torch.no_grad()
def build_base_pyramid(points: Tensor,
                        lengths: Tensor):
    """
    Only build the base of the graph pyramid, consisting of:
        > The subampled points for the first layer, in pack mode.
        > The lengths of the pack at first layer.
    """

    # Results lists
    pyramid = EasyDict()
    pyramid.points = []
    pyramid.lengths = []
    pyramid.neighbors = []
    pyramid.pools = []
    pyramid.upsamples = []
    pyramid.up_distances = []

    pyramid.points.append(points)
    pyramid.lengths.append(lengths)

    return pyramid


def fill_pyramid(pyramid: EasyDict,
                 num_layers: int,
                 sub_size: float,
                 search_radius: float,
                 radius_scaling: float,
                 neighbor_limits: List[int],
                 upsample_n: int = 1,
                 sub_mode: str = 'grid',
                 grid_pool_mode: bool = False):
    """
    Fill the graph pyramid, with:
        > The subampled points for each layer, in pack mode.
        > The lengths of the pack for each layer.
        > The neigbors indices for convolutions.
        > The pooling indices (neighbors from one layer to another).
        > The upsampling indices (opposite of pooling indices).
    """

    
    # Check if pyramid is already full
    if len(pyramid.neighbors) > 0:
        raise ValueError('Trying to fill a pyramid that already have neighbors')
    if len(pyramid.pools) > 0:
        raise ValueError('Trying to fill a pyramid that already have pools')
    if len(pyramid.upsamples) > 0:
        raise ValueError('Trying to fill a pyramid that already have upsamples')
    if len(pyramid.points) < 1:
        raise ValueError('Trying to fill a pyramid that does not have first points')
    if len(pyramid.lengths) < 1:
        raise ValueError('Trying to fill a pyramid that does not have first lengths')
    if len(pyramid.points) > 1:
        raise ValueError('Trying to fill a pyramid that already have more than one points')
    if len(pyramid.lengths) > 1:
        raise ValueError('Trying to fill a pyramid that already have more than one lengths')

    # Grid pool mode can only happen if method is grid
    grid_pool_mode = sub_mode == 'grid' and grid_pool_mode

    # Choose neighbor function depending on device
    if 'cuda' in pyramid.points[0].device.type:
        neighb_func = radius_search_pack_mode
    else:
        # neighb_func = batch_radius_neighbors
        neighb_func = batch_knn_neighbors

    # Subsample all point clouds on GPU
    points0 = pyramid.points[0]
    lengths0 = pyramid.lengths[0]
    for i in range(num_layers):

        if i > 0:

            if grid_pool_mode:
                sub_points, sub_lengths, poolings, upsamplings = batch_grid_partition(points0, lengths0, sub_size)
                pyramid.pools.append(poolings)
                pyramid.upsamples.append(upsamplings)
                points0 = sub_points
                lengths0 = sub_lengths


            else:
                sub_points, sub_lengths = subsample_pack_batch(points0, lengths0, sub_size, method=sub_mode)
                if sub_mode == 'fps':
                    points0 = sub_points
                    lengths0 = sub_lengths

            pyramid.points.append(sub_points)
            pyramid.lengths.append(sub_lengths)
            
        if sub_size > 0:
            sub_size *= radius_scaling


    # Find all neighbors
    for i in range(num_layers):

        # Get current points
        cur_points = pyramid.points[i]
        cur_lengths = pyramid.lengths[i]

        # Get convolution indices
        neighbors = neighb_func(cur_points, cur_points, cur_lengths, cur_lengths, search_radius, neighbor_limits[i])
        pyramid.neighbors.append(neighbors)

        # Relation with next layer 
        if not grid_pool_mode and i < num_layers - 1:
            sub_points = pyramid.points[i + 1]
            sub_lengths = pyramid.lengths[i + 1]

            # Get pooling indices
            subsampling_inds = neighb_func(sub_points, cur_points, sub_lengths, cur_lengths, search_radius, neighbor_limits[i])
            pyramid.pools.append(subsampling_inds)

            if upsample_n > 0:
                upsampling_inds, up_dists = neighb_func(cur_points, sub_points, cur_lengths, sub_lengths, search_radius * radius_scaling, upsample_n, return_dist=True)
                pyramid.upsamples.append(upsampling_inds)
                pyramid.up_distances.append(up_dists)


        # Increase radius for next layer
        search_radius *= radius_scaling

    # mean_dt = 1000 * (np.array(t[1:]) - np.array(t[:-1]))
    # message = ' ' * 2
    # for dt in mean_dt:
    #     message += ' {:5.1f}'.format(dt)
    # print(message)

    return


@torch.no_grad()
def build_full_pyramid(points: Tensor,
                       lengths: Tensor,
                       num_layers: int,
                       sub_size: float,
                       search_radius: float,
                       radius_scaling: float,
                       neighbor_limits: List[int],
                       upsample_n: int = 1,
                       sub_mode: str = 'grid',
                       grid_pool_mode: bool = False):
    """
    Build the graph pyramid, consisting of:
        > The subampled points for each layer, in pack mode.
        > The lengths of the pack.
        > The neigbors indices for convolutions.
        > The pooling indices (neighbors from one layer to another).
        > The upsampling indices (opposite of pooling indices).
    """

    pyramid = build_base_pyramid(points, lengths)

    fill_pyramid(pyramid,
                 num_layers,
                 sub_size,
                 search_radius,
                 radius_scaling,
                 neighbor_limits,
                 upsample_n,
                 sub_mode,
                 grid_pool_mode)

    return pyramid


@torch.no_grad()
def pyramid_neighbor_stats(points: Tensor,
                           num_layers: int,
                           sub_size: float,
                           search_radius: float,
                           radius_scaling: float,
                           sub_mode: str = 'grid'):
    """
    Function used for neighbors calibration. Return the average number of neigbors at each layer.
    Args:
        points (Tensor): initial layer points (M, 3).
        num_layers (int): number of layers.
        sub_size (float): initial subsampling size
        radius (float): search radius.
        sub_mode (str): the subsampling method ('grid', 'ph', 'fps').
    Returns:
        counts_list (List[Tensor]): All neigbors counts at each layers
    """

    counts_list = []
    lengths = [points.shape[0]]
    for i in range(num_layers):
        if i > 0:
            points, lengths = subsample_pack_batch(points, lengths, sub_size, method=sub_mode)
        counts = keops_radius_count(points, points, search_radius)
        counts_list.append(counts)
        if sub_size > 0:
            sub_size *= radius_scaling
        search_radius *= radius_scaling
    return counts_list