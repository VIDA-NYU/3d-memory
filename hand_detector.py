# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
import _init_paths
import os
import numpy as np
import cv2
import torch

from model.utils.config import cfg, cfg_from_file
from model.rpn.bbox_transform import clip_boxes
from model.roi_layers import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.blob import im_list_to_blob
from model.faster_rcnn.resnet import resnet

cfg_file_path = './hand_object_detector/cfgs/res101.yml'
model_dir = './hand_object_detector/models/res101_handobj_100K/pascal_voc'

def _get_image_blob(im):
  """Converts an image into a network input.
  Arguments:
    im (ndarray): a color image in BGR order
  Returns:
    blob (ndarray): a data blob holding an image pyramid
    im_scale_factors (list): list of image scales (relative to im) used
      in the image pyramid
  """
  im_orig = im.astype(np.float32, copy=True)
  im_orig -= cfg.PIXEL_MEANS

  im_shape = im_orig.shape
  im_size_min = np.min(im_shape[0:2])
  im_size_max = np.max(im_shape[0:2])

  processed_ims = []
  im_scale_factors = []

  for target_size in cfg.TEST.SCALES:
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
      im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
            interpolation=cv2.INTER_LINEAR)
    im_scale_factors.append(im_scale)
    processed_ims.append(im)

  # Create a blob to hold the input images
  blob = im_list_to_blob(processed_ims)

  return blob, np.array(im_scale_factors)

class HandDetector:
  def __init__(self):
    self.cuda_flag = True
    self.class_agnostic = False
    self.thresh_hand = 0.5
    self.thresh_obj = 0.5

    cfg_from_file(cfg_file_path)
    cfg.USE_GPU_NMS = self.cuda_flag
    np.random.seed(cfg.RNG_SEED)

    # load model
    if not os.path.exists(model_dir):
      raise Exception('There is no input directory for loading network from ' + model_dir)
    load_name = os.path.join(model_dir, 'faster_rcnn_{}_{}_{}.pth'.format(1, 8, 132028))

    self.pascal_classes = np.asarray(['__background__', 'targetobject', 'hand']) 

    # initilize the network here.
    self.fasterRCNN = resnet(self.pascal_classes, 101, pretrained=False, class_agnostic=self.class_agnostic)
    self.fasterRCNN.create_architecture()

    print("load checkpoint %s" % (load_name))
    if self.cuda_flag > 0:
      checkpoint = torch.load(load_name)
    else:
      checkpoint = torch.load(load_name, map_location=(lambda storage, loc: storage))
    self.fasterRCNN.load_state_dict(checkpoint['model'])
    if 'pooling_mode' in checkpoint.keys():
      cfg.POOLING_MODE = checkpoint['pooling_mode']
    print('load model successfully!')


    # initilize the tensor holder here.
    self.im_data = torch.FloatTensor(1)
    self.im_info = torch.FloatTensor(1)
    self.num_boxes = torch.LongTensor(1)
    self.gt_boxes = torch.FloatTensor(1)
    self.box_info = torch.FloatTensor(1) 

    # ship to cuda
    if self.cuda_flag > 0:
      self.im_data = self.im_data.cuda()
      self.im_info = self.im_info.cuda()
      self.num_boxes = self.num_boxes.cuda()
      self.gt_boxes = self.gt_boxes.cuda()

    with torch.no_grad():
      if self.cuda_flag > 0:
        cfg.CUDA = True
      if self.cuda_flag > 0:
        self.fasterRCNN.cuda()
      self.fasterRCNN.eval()

  def predict(self, im_in):
    # bgr
    im = im_in

    blobs, im_scales = _get_image_blob(im)
    assert len(im_scales) == 1, "Only single-image batch implemented"
    im_blob = blobs
    im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

    im_data_pt = torch.from_numpy(im_blob)
    im_data_pt = im_data_pt.permute(0, 3, 1, 2)
    im_info_pt = torch.from_numpy(im_info_np)

    with torch.no_grad():
      self.im_data.resize_(im_data_pt.size()).copy_(im_data_pt)
      self.im_info.resize_(im_info_pt.size()).copy_(im_info_pt)
      self.gt_boxes.resize_(1, 1, 5).zero_()
      self.num_boxes.resize_(1).zero_()
      self.box_info.resize_(1, 1, 5).zero_() 

    rois, cls_prob, bbox_pred, \
    rpn_loss_cls, rpn_loss_box, \
    RCNN_loss_cls, RCNN_loss_bbox, \
    rois_label, loss_list = self.fasterRCNN(self.im_data, self.im_info, self.gt_boxes, self.num_boxes, self.box_info) 

    scores = cls_prob.data
    boxes = rois.data[:, :, 1:5]

    # extact predicted params
    contact_vector = loss_list[0][0] # hand contact state info
    offset_vector = loss_list[1][0].detach() # offset vector (factored into a unit vector and a magnitude)
    lr_vector = loss_list[2][0].detach() # hand side info (left/right)

    # get hand contact 
    _, contact_indices = torch.max(contact_vector, 2)
    contact_indices = contact_indices.squeeze(0).unsqueeze(-1).float()

    # get hand side 
    lr = torch.sigmoid(lr_vector) > 0.5
    lr = lr.squeeze(0).float()

    if cfg.TEST.BBOX_REG:
        # Apply bounding-box regression deltas
        box_deltas = bbox_pred.data
        if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
        # Optionally normalize targets by a precomputed mean and stdev
          if self.class_agnostic:
              if self.cuda_flag > 0:
                  box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                            + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
              else:
                  box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                            + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)

              box_deltas = box_deltas.view(1, -1, 4)
          else:
              if self.cuda_flag > 0:
                  box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                            + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
              else:
                  box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                            + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
              box_deltas = box_deltas.view(1, -1, 4 * len(self.pascal_classes))

        pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
        pred_boxes = clip_boxes(pred_boxes, self.im_info.data, 1)
    else:
        # Simply repeat the boxes, once for each class
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    pred_boxes /= im_scales[0]

    scores = scores.squeeze()
    pred_boxes = pred_boxes.squeeze()

    obj_dets, hand_dets = None, None
    for j in range(1, len(self.pascal_classes)):
        # inds = torch.nonzero(scores[:,j] > thresh).view(-1)
        if self.pascal_classes[j] == 'hand':
          inds = torch.nonzero(scores[:,j]>self.thresh_hand).view(-1)
        elif self.pascal_classes[j] == 'targetobject':
          inds = torch.nonzero(scores[:,j]>self.thresh_obj).view(-1)

        # if there is det
        if inds.numel() > 0:
          cls_scores = scores[:,j][inds]
          _, order = torch.sort(cls_scores, 0, True)
          if self.class_agnostic:
            cls_boxes = pred_boxes[inds, :]
          else:
            cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]
          
          cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1), contact_indices[inds], offset_vector.squeeze(0)[inds], lr[inds]), 1)
          cls_dets = cls_dets[order]
          keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
          cls_dets = cls_dets[keep.view(-1).long()]
          if self.pascal_classes[j] == 'targetobject':
            obj_dets = cls_dets.cpu().numpy()
          if self.pascal_classes[j] == 'hand':
            hand_dets = cls_dets.cpu().numpy()
          
    return obj_dets, hand_dets
