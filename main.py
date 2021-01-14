#!/usr/bin/python

import argparse
import torch, torchvision

# Setup detectron2 logger
import detectron2

# import some common libraries
import numpy as np
import os, cv2

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image_dir", required=True, help="Raw images directory")
    ap.add_argument("-o", "--output_dir", required=True, help="Output instance segmentation directory")
    ap.add_argument("-c", "--output_color_dir", default=False, help="Output color segmentation directory")
    args = vars(ap.parse_args())

    # List of image to be processed
    images_list = os.listdir(args["image_dir"])
    
    # Config Detectron2
    cfg = get_cfg()
    # Adding project-specific config
    cfg.merge_from_file(model_zoo.get_config_file("Cityscapes/mask_rcnn_R_50_FPN.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo.
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("Cityscapes/mask_rcnn_R_50_FPN.yaml")
    # Creating the predictor
    predictor = DefaultPredictor(cfg)

    # For each of the images in the directory
    for image in images_list:
        # Read the image
        im = cv2.imread(args["image_dir"]+'/'+image)
        print(image)
        # Prediction
        outputs = predictor(im)

        # Color visualization
        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.0)
        out_viz = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        color_im = out_viz.get_image()[:, :, ::-1]
        os.chdir(args["output_color_dir"]) 
        cv2.imwrite(image, color_im)

        # Vector of labels in the image to keep track of the instances
        n_labels = np.unique(outputs["instances"].pred_classes.cpu().numpy(), axis=0)
        lb_inst = dict(zip(n_labels, np.zeros_like(n_labels)))

        # scores sorted 
        Scores = outputs["instances"].pred_classes.cpu().numpy()
        idx_s_dict = {k: v for k, v in enumerate(Scores)}
        sorted_scores = {k: v for k, v in sorted(idx_s_dict.items(), key=lambda item: item[1], reverse=True)}

        # Output image
        Out_im = np.zeros_like(im[:, :, 0])

        # Processing the instances within the image
        for key in sorted_scores:
            # Mask and label of the instance
            Mask  = outputs["instances"].pred_masks[key].cpu().numpy()
            label = int(outputs["instances"].pred_classes[key].cpu().numpy())

            # encoding label and instance in the same number
            aux_mask = Mask*((label+1) * 1000 + lb_inst[label])
            aux_mask = (Out_im == 0)*aux_mask
            Out_im = np.add(Out_im, aux_mask)

            # incrementing the instance index for the corresponding label 
            lb_inst[label] += 1.0


        os.chdir(args["output_dir"])
        cv2.imwrite(image, Out_im.astype(np.uint16))


