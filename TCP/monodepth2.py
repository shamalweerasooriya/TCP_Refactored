# monodepth_module.py

import torch
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional
import os
from TCP.monodepth2network import *
from torchvision import transforms

class MonodepthModel:
    def __init__(self, model_name: str = "mono_640x192", use_gpu: bool = False):

        map_location = "cuda" if use_gpu else "cpu"

        # Download the model if it doesn't exist
        download_model_if_doesnt_exist(model_name)
        encoder_path = os.path.join("models", model_name, "encoder.pth")
        depth_decoder_path = os.path.join("models", model_name, "depth.pth")

        # Create the ResNet Encoder and Depth Decoder instances
        self.encoder = ResnetEncoder(18, False)
        self.depth_decoder = DepthDecoder(num_ch_enc=self.encoder.num_ch_enc, scales=range(4))

        # Load the pre-trained weights into the models
        self.loaded_dict_enc = torch.load(encoder_path, map_location=map_location)
        filtered_dict_enc = {k: v for k, v in self.loaded_dict_enc.items() if k in self.encoder.state_dict()}
        self.encoder.load_state_dict(filtered_dict_enc)

        loaded_dict = torch.load(depth_decoder_path, map_location=map_location)
        self.depth_decoder.load_state_dict(loaded_dict)

        # Set the models to evaluation mode
        self.encoder.eval()
        self.depth_decoder.eval()

        # Move the models to the GPU if available and specified
        if use_gpu and torch.cuda.is_available():
            self.encoder = self.encoder.cuda()
            self.depth_decoder = self.depth_decoder.cuda()

    def predict_depth_batch(self, input_images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            # Prepare a list to store the features for each image in the batch
            features_list = []
            depth_feature_list = []
            for i in range(32): # TODO - this shouldn't be hard coded
                feed_height = self.loaded_dict_enc['height']
                feed_width = self.loaded_dict_enc['width']
                resized_tensor = F.interpolate(input_images[i].unsqueeze(0), size=(feed_height, feed_width), mode='bicubic', align_corners=False)
                features = self.encoder(resized_tensor)
                depth_feature = self.depth_decoder(features)
                features_list.append(features[-1])
                depth_feature_list.append(depth_feature[("disp", 0)])
        
        squeezed_features = torch.cat(features_list, dim=0).squeeze()
        squeezed_depth_features = torch.cat(depth_feature_list, dim=0).squeeze()


        return squeezed_features, squeezed_depth_features