# class KPConvSingleKernelPoint(nn.Module):
#   def __init__(
#     self,
#     kernel: torch.Tensor,
#     neighbor_count: int,
#     input_feature_count: int,
#     output_feature_count: int,
#   ):
#     super().__init__()

#     self.neighbor_count = neighbor_count

#     # (K, d)
#     self.kernel = kernel
#     self.kernel_tree = KDTree(kernel)

#     # (K, C, O)
#     v = torch.randn(self.kernel.size(0), input_feature_count, output_feature_count)
#     v[...] = -1.0
#     v[self.kernel[:, 0] > 0, ...] = 1.0
#     # self.weights = nn.Parameter(torch.randn(self.kernel.size(0), input_feature_count, output_feature_count))
#     self.weights = nn.Parameter(v)

#   # G = number of support points
#   # Q = number of query points
#   # K = number of kernel points
#   def forward(
#     self,
#     query_points: torch.Tensor, # (Q, d)
#     support_points: torch.Tensor, # (G, d)
#     support_features: torch.Tensor, # (G, C)
#   ):
#     print(f'Q={query_points.size(0)}')
#     print(f'G={support_points.size(0)}')
#     print(f'K={self.kernel.size(0)}')
#     print(f'C={support_features.size(1)}')
#     print(f'O={self.weights.size(2)}')
#     print(f'H={self.neighbor_count}')

#     d = query_points.size(1)

#     tree = KDTree(support_points)

#     # Shape: (len(query_points), neighbor_count)
#     # Shape: (Q, H)
#     neighbor_indices = tree.query(query_points, k=self.neighbor_count, return_distance=False)

#     # Shape: (Q, H, d)
#     neighbor_points = query_points[neighbor_indices, :]

#     # Shape: (Q, H, C)
#     neighbor_features = support_features[neighbor_indices, :]

#     # (Q, H, 1)
#     diff = neighbor_points - query_points[:, None, :]
#     distances, kernel_point_indices = self.kernel_tree.query(diff.view(-1, d), k=1)

#     # (Q, H)
#     distances = torch.tensor(distances.reshape(neighbor_indices.shape))
#     kernel_point_indices = torch.tensor(kernel_point_indices.reshape(neighbor_indices.shape))

#     # (Q, H)
#     sigma = 1.0
#     influences = torch.relu(1 - distances / sigma)

#     return np.einsum('qh, qhco, qhc -> qo', influences, self.weights[kernel_point_indices, :, :], neighbor_features)

#     # distances = torch.sqrt(((neighbor_points[:, :, None, :] - query_points[:, None, None, :] - self.kernel) ** 2).sum(dim=-1))

#     # # (Q, H, K)
#     # sigma = 1.0
#     # influences = torch.relu(1 - distances / sigma)

#     # return np.einsum('qhk, kco, qhc -> qo', influences, self.weights, neighbor_features)
