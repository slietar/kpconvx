#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
# ----------------------------------------------------------------------------------------------------------------------
#
#   Hugues THOMAS - 06/10/2023
#
#   KPConvX project: cuda_funcs.py
#       > FPS implementation from from PointNet++ (for debug)
#


# Cuda extensions
# import cpp_wrappers.pointnet2_batch.pointnet2_batch_cuda as pointnet2_cuda

import os
import torch
import numpy as np


@torch.no_grad()
def furthest_point_sample(points, new_n=None, stride=4, min_d=0):
    """
    Fursthest sampling controlled by the new number of points wanted. Single point cloud.
    Args:
        points (Tensor): the original points (N, D).
        new_n     (int): the new number of points wanted .
        stride    (int): If new_n is not provided, N is divided by stride
    Returns:
        sample_inds (LongTensor): the indices of the subsampled points (M,).
    """
    
    # Verify contiguous
    assert points.is_contiguous()

    # Get dimensions
    N = points.shape[0]
    B = 1
    points = points.view(B, N, -1)

    # Create new_lengths if not provided
    if new_n is None:
        new_n = int(np.floor(N / stride))

    idx = torch.cuda.IntTensor(B, new_n, device=points.device).fill_(-1)
    temp = torch.cuda.FloatTensor(B, N, device=points.device).fill_(1e10)
    pointnet2_cuda.furthest_point_sampling_wrapper(B, N, new_n, min_d, points, temp, idx)
    del temp
    idx = idx.view(-1).long()

    # Remove -1 indices as they are just ignored values
    return idx[idx > -1]


@torch.no_grad()
def furthest_point_sample_3(points, new_n=None, stride=4):
    """
    Naive Fursthest sampling
    Args:
        points (Tensor): the original points (N, D).
        new_n     (int): the new number of points wanted .
        stride    (int): If new_n is not provided, N is divided by stride
    Returns:
        sample_inds (LongTensor): the indices of the subsampled points (M,).
    """

    # In case no batch
    no_batch = points.ndim < 3
    if no_batch:
        points = points.unsqueeze(0)

    # Dimensions
    device = points.device
    B, N, C = points.shape
    
    # Create new_lengths if not provided
    if new_n is None:
        new_n = N // stride

    centroids = torch.zeros(B, new_n, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(new_n):
        centroids[:, i] = farthest
        centroid = points[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((points - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
        
    if no_batch:
        centroids = centroids.squeeze()
    return centroids
