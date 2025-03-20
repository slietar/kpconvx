from pathlib import Path
from typing import Any

import matplotlib
import torch
from data_handlers.scene_seg import SceneSegCollate, SceneSegSampler
from experiments.S3DIS.S3DIS_rooms import S3DIRDataset
from models.InvolutionNet import InvolutionFCNN
from models.KPConvNet import KPFCNN as KPConvFCNN
from models.KPInvNet import KPInvFCNN
from models.KPNext import KPNeXt
from ply import write_ply
from torch.utils.data import DataLoader
from utils.config import load_cfg


if __name__ == '__main__':
  root_path = Path('../..')
  log_path = root_path / 'S3DIS_KPConvD-L'
  weights_path = log_path / 'checkpoints/current_chkp.tar'


  #############
  # Load config
  #############

  # Configuration parameters
  cfg: Any = load_cfg(str(log_path))
  cfg.data.path = str(root_path / 's3dis')


  ###################
  # Define parameters
  ###################

  # Optionally you can change some parameters from the config file. For example:

  # Ensure we only have one point cloud in each batch for test
  cfg.test.batch_limit = 1

  # Test whole rooms
  cfg.test.in_radius = 100.0

  # Only stop one test epoch when all rooms have been tested
  cfg.test.max_steps_per_epoch = 9999999

  # Test 10 times
  cfg.test.max_votes = 10

  # Augmentations
  cfg.augment_test.anisotropic = False
  cfg.augment_test.scale = [0.99, 1.01]
  cfg.augment_test.flips = [0.5, 0, 0]
  cfg.augment_test.rotations = 'vertical'
  cfg.augment_test.jitter = 0
  cfg.augment_test.color_drop = 0.0
  cfg.augment_test.chromatic_contrast = False
  cfg.augment_test.chromatic_all = False



  test_dataset = S3DIRDataset(cfg,
                              chosen_set='test',
                              precompute_pyramid=True)

  # Calib from training data
  # test_dataset.calib_batch(new_cfg)
  # test_dataset.calib_neighbors(new_cfg)

  # Initialize samplers
  test_sampler = SceneSegSampler(test_dataset)
  test_loader = DataLoader(test_dataset,
                            batch_size=1,
                            sampler=test_sampler,
                            collate_fn=SceneSegCollate,
                            num_workers=1,
                            pin_memory=True)


  ###############
  # Build network
  ###############


  modulated = False
  if 'mod' in cfg.model.kp_mode:
      modulated = True

  if cfg.model.kp_mode in ['kpconvx', 'kpconvd', 'kpconv', 'kpconvtest']:
      net = KPNeXt(cfg)

  elif cfg.model.kp_mode.startswith('kpconv') or cfg.model.kp_mode.startswith('kpmini'):
      net = KPConvFCNN(cfg, modulated=modulated, deformable=False)
  elif cfg.model.kp_mode.startswith('kpdef'):
      net = KPConvFCNN(cfg, modulated=modulated, deformable=True)
  elif cfg.model.kp_mode.startswith('kpinv'):
      net = KPInvFCNN(cfg)
  elif cfg.model.kp_mode.startswith('transformer') or cfg.model.kp_mode.startswith('inv_'):
      net = InvolutionFCNN(cfg)
  elif cfg.model.kp_mode.startswith('kpnext'):
      net = KPNeXt(cfg, modulated=modulated, deformable=False)
  else:
      raise ValueError('Unknown network mode')


  #########################
  # Load pretrained weights
  #########################


  # Load previous checkpoint
  checkpoint = torch.load(weights_path, map_location=torch.device('cpu'))
  net.load_state_dict(checkpoint['model_state_dict'])
  net.eval()

  ############
  # Start test
  ############

  test_path = log_path / 'test'
  test_path.mkdir(exist_ok=True)



  device = torch.device('cpu')

  ####################
  # Initialize network
  ####################

  # Get the network to the device we chose
  net.to(device)
  net.eval()



  # Choose validation smoothing parameter (0 for no smothing, 0.99 for big smoothing)
  test_smooth = cfg.test.test_momentum
  softmax = torch.nn.Softmax(1)

  # Number of classes predicted by the model
  nc_model = net.num_logits

  #####################
  # Network predictions
  #####################

  batch = next(iter(test_loader))

#   print([p.size() for p in batch.in_dict.points])

  with torch.no_grad():
    pred = net(batch.to(device)).cpu().numpy()

#   print(pred.size())
#   print(pred[0, :])
#   print(pred[1, :])
#   print(pred[2, :])

  colors = matplotlib.colormaps['tab20'](pred.argmax(axis=1) % 20)[:, :3]

  write_ply(str(log_path / 'test.ply'), [batch.in_dict.points[0].cpu().numpy(), colors], ['x', 'y', 'z', 'red', 'green', 'blue'])
