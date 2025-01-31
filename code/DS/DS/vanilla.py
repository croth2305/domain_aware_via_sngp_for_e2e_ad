from collections import deque
import numpy as np
import torch 
from torch import nn
from DS.resnet import *
import math
import torch.nn.functional as F

from ignite.engine import Events, Engine
from ignite.metrics import Average, Loss
from ignite.contrib.handlers import ProgressBar
import gpytorch

class FCResNet(nn.Module):
    def __init__(
        self,
        #input_dim,
        features,
        #depth,
        spectral_normalization,
        coeff=0.95,
        n_power_iterations=1,
        dropout_rate=0.01,
        num_outputs=None,
        activation="relu",
    ):
        super().__init__()
        dims_in = [3, 24, 36, 48, 64]
        dims_out = [24, 36, 48, 64, 64]
        kernel = [5, 5, 5, 3, 3]
        stride = [2, 2, 2, 1, 1]
        dims = [
            (3, 256, 900), 
            (24, 127, 449), 
            (36, 64, 225), 
            (48, 32, 113), 
            (64, 32, 113),
            (64, 32, 113)]
        
        self.first = nn.Conv2d(
            dims_in[0], 
            dims_out[0], 
            kernel_size=kernel[0], 
            stride=stride[0],
            padding=1)
        
        self.res_second = nn.ModuleList(
            [
                nn.Conv2d(
                    dims_in[1],
                    dims_out[1],
                    kernel_size=kernel[1], 
                    stride=stride[1],
                    padding=2
                ),
                nn.Conv2d(
                    dims_out[1],
                    dims_out[1],
                    kernel_size=kernel[1], 
                    stride=1,
                    padding=2
                ),
                #downsampling
                nn.Conv2d(
                    dims_in[1],
                    dims_out[1],
                    kernel_size=1, 
                    stride=stride[1],
                    padding=0
                )
            ]
        )

        self.res_third = nn.ModuleList(
            [
                nn.Conv2d(
                    dims_in[2],
                    dims_out[2],
                    kernel_size=kernel[2], 
                    stride=stride[2],
                    padding=2
                ),
                nn.Conv2d(
                    dims_out[2],
                    dims_out[2],
                    kernel_size=kernel[2], 
                    stride=1,
                    padding=2
                ),
                #downsampling
                nn.Conv2d(
                    dims_in[2],
                    dims_out[2],
                    kernel_size=1, 
                    stride=stride[2],
                    padding=0
                )
            ]
        )
        
        self.res_fourth = nn.ModuleList(
            [
                nn.Conv2d(
                    dims_in[3],
                    dims_out[3],
                    kernel_size=kernel[3], 
                    stride=stride[3],
                    padding=1
                ),
                nn.Conv2d(
                    dims_out[3],
                    dims_out[3],
                    kernel_size=kernel[3], 
                    stride=1,
                    padding=1
                ),
                #downsampling
                nn.Conv2d(
                    dims_in[3],
                    dims_out[3],
                    kernel_size=1, 
                    stride=stride[3],
                    padding=0
                )
            ]
        )

        self.res_fifth = nn.ModuleList(
            [
                nn.Conv2d(
                    dims_in[4],
                    dims_out[4],
                    kernel_size=kernel[4], 
                    stride=stride[4],
                    padding=1
                ),
                nn.Conv2d(
                    dims_out[4],
                    dims_out[4],
                    kernel_size=kernel[4], 
                    stride=1,
                    padding=1
                ),
                #downsampling
                nn.Conv2d(
                    dims_in[4],
                    dims_out[4],
                    kernel_size=1, 
                    stride=stride[4],
                    padding=0
                )
            ]
        )
        
        self.reduce = nn.Linear(231424, 1024)
        #self.reduce = nn.Linear(231424, 1024+256)
  
        self.first_m = nn.Linear(9, 128)
        self.measurement_encoder = nn.ModuleList([
            nn.Linear(128, 256),
            nn.Linear(256, 256),
            #upsampling
            nn.Linear(128, 256)
        ])

        self.dropout = nn.Dropout(dropout_rate)

        if activation == "relu":
            self.activation = F.relu
        elif activation == "elu":
            self.activation = F.elu
        else:
            raise ValueError("That acivation is unknown")

    def forward(self, front_img, state):
        # image encoder
        encoded_i = self.first(front_img)

        i = encoded_i
        out = self.dropout(self.activation(self.res_second[0](encoded_i)))
        out = self.res_second[1](out)
        i = self.res_second[2](encoded_i)
        out += i
        out = self.activation(out)

        encoded_i = out
        i = encoded_i
        out = self.dropout(self.activation(self.res_third[0](encoded_i)))
        out = self.res_third[1](out)
        i = self.res_third[2](encoded_i)
        out += i
        out = self.activation(out)

        encoded_i = out
        i = encoded_i
        out = self.dropout(self.activation(self.res_fourth[0](encoded_i)))
        out = self.res_fourth[1](out)
        i = self.res_fourth[2](encoded_i)
        out += i
        out = self.activation(out) 

        encoded_i = out
        i = encoded_i
        out = self.dropout(self.activation(self.res_fifth[0](encoded_i)))
        out = self.res_fifth[1](out)
        i = self.res_fifth[2](encoded_i)
        out += i
        out = self.activation(out)
        map = out

        encoded_i = out.view(out.size(0), -1)  # Flatten the tensor
        features = self.reduce(encoded_i)

        # measurement encoder
        encoded_m = self.first_m(state)
        i = encoded_m
        out = self.dropout(self.activation(self.measurement_encoder[0](encoded_m)))
        out = self.measurement_encoder[1](out)
        i = self.measurement_encoder[2](encoded_m)
        out += i
        encoded_m = self.activation(out)

        # add state to encoded image
        features = torch.cat([features, encoded_m], dim=1)
        
        return features, map

class DAVE2_VANILLA(nn.Module):

    def __init__(
        self,
        feature_extractor,
        num_deep_features,
        num_gp_features,
        normalize_gp_features,
        num_random_features,
        num_outputs,
        num_data,
        train_batch_size,
        ridge_penalty=1.0,
        feature_scale=None,
        mean_field_factor=None,  # required for classification problems
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.mean_field_factor = mean_field_factor
        self.ridge_penalty = ridge_penalty
        self.train_batch_size = train_batch_size

        if num_gp_features > 0:
            self.num_gp_features = num_gp_features
            self.register_buffer(
                "random_matrix",
                torch.normal(0, 0.05, (num_gp_features, num_deep_features)),
            )
            self.jl = lambda x: nn.functional.linear(x, self.random_matrix)
        else:
            self.num_gp_features = num_deep_features
            self.jl = nn.Identity()

        self.normalize_gp_features = normalize_gp_features
        if normalize_gp_features:
            self.normalize_steer = nn.LayerNorm(num_gp_features)
            self.normalize_accel = nn.LayerNorm(num_gp_features)

        self.rff_steer = nn.Linear(
            num_gp_features, num_random_features)
        self.rff_accel = nn.Linear(
            num_gp_features, num_random_features)

        self.beta_steer = nn.Linear(num_random_features, 1)
        self.beta_accel = nn.Linear(num_random_features, 2)

    def forward(self, front_img, state=None):
        outputs = {}
        f, m = self.feature_extractor(front_img, state)
        #print(f.shape) # (128, (1024+256) 1280)
        outputs['feature_map'] = m
        f_reduc = self.jl(f)
        if self.normalize_gp_features:
            f_reduc = self.normalize_steer(f_reduc)
            f_reduc = self.normalize_accel(f_reduc)

        k_steer = self.rff_steer(f_reduc)
        pred_steer = self.beta_steer(k_steer)
        #pred_steer = self.tanh(pred_steer)
        pred_steer = torch.tanh(pred_steer)
        outputs['pred_steer'] = pred_steer 

        k_accel = self.rff_accel(f_reduc)
        pred_ab = self.beta_accel(k_accel)
        #pred_accel = self.tanh(pred_accel)
        pred_accel = (torch.tanh(pred_ab[:, 0])+1)/2
        pred_brake = (torch.tanh(pred_ab[:, 1])+1)/2
        outputs['pred_accel'] = pred_accel 
        outputs['pred_brake'] = pred_brake

        return outputs
    