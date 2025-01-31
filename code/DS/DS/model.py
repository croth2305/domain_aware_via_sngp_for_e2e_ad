from collections import deque
import numpy as np
import torch 
from torch import nn
from DS.resnet import *

from due.layers import spectral_norm_fc
import torch.nn.functional as F
from torch.nn.functional import normalize, conv_transpose2d, conv2d
from torch.nn.utils.spectral_norm import (
    SpectralNorm,
    SpectralNormLoadStateDictPreHook,
    SpectralNormStateDictHook,
)
import math

import gpytorch
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import RBFKernel, RQKernel, MaternKernel, ScaleKernel
from gpytorch.means import ConstantMean
from gpytorch.models import ApproximateGP

from gpytorch.variational import (
    CholeskyVariationalDistribution,
    IndependentMultitaskVariationalStrategy,
    VariationalStrategy,
)

from sklearn import cluster

class SpectralNormConv(SpectralNorm):
    def compute_weight(self, module, do_power_iteration: bool) -> torch.Tensor:
        weight = getattr(module, self.name + "_orig")
        u = getattr(module, self.name + "_u")
        v = getattr(module, self.name + "_v")

        # get settings from conv-module (for transposed convolution parameters)
        stride = module.stride
        padding = module.padding

        if do_power_iteration:
            with torch.no_grad():
                output_padding = 0
                if stride[0] > 1:
                    # Note: the below does not generalize to stride > 2
                    output_padding = 1 - self.input_dim[-1] % 2
                #print(f"s:{stride[0]}, p:{padding} , op: {output_padding}")
                for _ in range(self.n_power_iterations):
                    v_s = conv_transpose2d(
                        u.view(self.output_dim),
                        weight,
                        stride=stride,
                        padding=padding,
                        output_padding=output_padding,
                    )

                    if not (v.shape == v_s.view(-1).shape):
                        out_h = abs(v_s.shape[2] - self.input_dim[2])
                        out_w = abs(v_s.shape[3] - self.input_dim[3])
                        if stride[0] == 1:
                            padding = 0 
                        
                        if out_w == out_h:
                            padding += int(out_w/2)
                            out_w = 0
                            out_h = 0
                        

                        v_s = conv_transpose2d(
                            u.view(self.output_dim),
                            weight,
                            stride=stride,
                            padding=padding,
                            output_padding=(
                                abs(output_padding-out_h), 
                                abs(output_padding-out_w)),
                        )
                        
                        if not (v.shape == v_s.view(-1).shape):
                            padding = 0
                            v_s = conv_transpose2d(
                                u.view(self.output_dim),
                                weight,
                                stride=stride,
                                padding=padding,
                                output_padding=(
                                    abs(output_padding-out_h), 
                                    abs(output_padding-out_w)),
                                )

                    v = normalize(v_s.view(-1), dim=0, eps=self.eps, out=v)
                    
                    try:
                        u_s = conv2d(
                            v.view(self.input_dim),
                            weight,
                            stride=stride,
                            padding=padding,
                            bias=None,
                        )
                    except:
                        v_s = conv_transpose2d(
                            u.view(self.output_dim),
                            weight,
                            stride=stride,
                            padding=padding,
                            output_padding=(0,output_padding) if output_padding > 0 else (1,0),
                        )
                        v = normalize(v_s.view(-1), dim=0, eps=self.eps, out=v)
                        u_s = conv2d(
                            v.view(self.input_dim),
                            weight,
                            stride=stride,
                            padding=padding,
                            bias=None,
                        )

                    #print(f"u shape: {u.shape}, u_s shape: {u_s.shape}")
                    u = normalize(u_s.view(-1), dim=0, eps=self.eps, out=u)
                if self.n_power_iterations > 0:
                    # See above on why we need to clone
                    u = u.clone(memory_format=torch.contiguous_format)
                    v = v.clone(memory_format=torch.contiguous_format)

        weight_v = conv2d(
            v.view(self.input_dim), weight, stride=stride, padding=padding, bias=None
        )
        weight_v = weight_v.view(-1)
        sigma = torch.dot(u.view(-1), weight_v)
        # soft normalization: only when sigma larger than coeff
        factor = torch.max(torch.ones(1, device=weight.device), sigma / self.coeff)
        weight = weight / factor

        # for logging
        sigma_log = getattr(module, self.name + "_sigma")
        sigma_log.copy_(sigma.detach())

        return weight

    def __call__(self, module, inputs):
        assert (
            inputs[0].shape[1:] == self.input_dim[1:]
        ), f"Input dims don't match actual input {inputs[0].shape[1:]}, {self.input_dim[1:]}"
        setattr(
            module,
            self.name,
            self.compute_weight(module, do_power_iteration=module.training),
        )

    @staticmethod
    def apply(module, coeff, input_dim, name, n_power_iterations, eps):
        for k, hook in module._forward_pre_hooks.items():
            if isinstance(hook, SpectralNormConv) and hook.name == name:
                raise RuntimeError(
                    "Cannot register two spectral_norm hooks on "
                    "the same parameter {}".format(name)
                )

        fn = SpectralNormConv(name, n_power_iterations, eps=eps)
        fn.coeff = coeff
        fn.input_dim = input_dim
        weight = module._parameters[name]
        #print(f"weight shape: {weight.shape}")

        with torch.no_grad():
            num_input_dim = input_dim[0] * input_dim[1] * input_dim[2] * input_dim[3]
            v = normalize(torch.randn(num_input_dim), dim=0, eps=fn.eps)

            # get settings from conv-module (for transposed convolution)
            stride = module.stride
            padding = module.padding
            # forward call to infer the shape
            u = conv2d(
                v.view(input_dim), weight, stride=stride, padding=padding, bias=None
            )
            fn.output_dim = u.shape
            #print(f"output_dim: {fn.output_dim}")
            num_output_dim = (
                fn.output_dim[0]
                * fn.output_dim[1]
                * fn.output_dim[2]
                * fn.output_dim[3]
            )
            # overwrite u with random init
            u = normalize(torch.randn(num_output_dim), dim=0, eps=fn.eps)

        #print(f"u shape: {u.shape}, v shape: {v.shape}, input_dim: {input_dim}, output_dim: {fn.output_dim}")
        delattr(module, fn.name)
        module.register_parameter(fn.name + "_orig", weight)
        setattr(module, fn.name, weight.data)
        module.register_buffer(fn.name + "_u", u)
        module.register_buffer(fn.name + "_v", v)
        module.register_buffer(fn.name + "_sigma", torch.ones(1))

        module.register_forward_pre_hook(fn)

        module._register_state_dict_hook(SpectralNormStateDictHook(fn))
        module._register_load_state_dict_pre_hook(SpectralNormLoadStateDictPreHook(fn))
        return fn

def spectral_norm_conv(
    module, coeff, input_dim, n_power_iterations=1, name="weight", eps=1e-12,
):
    """
    Applies spectral normalization to Convolutions with flexible max norm

    Args:
        module (nn.Module): containing convolution module
        input_dim (tuple(int, int, int)): dimension of input to convolution
        coeff (float, optional): coefficient to normalize to
        n_power_iterations (int, optional): number of power iterations to
            calculate spectral norm
        name (str, optional): name of weight parameter
        eps (float, optional): epsilon for numerical stability in
            calculating norms

    Returns:
        The original module with the spectral norm hook

    Example::

        >>> m = spectral_norm_conv(nn.Conv2D(3, 16, 3), (3, 32, 32), 2.0)

    """

    input_dim_4d = torch.Size([1, input_dim[0], input_dim[1], input_dim[2]])
    SpectralNormConv.apply(module, coeff, input_dim_4d, name, n_power_iterations, eps)

    return module

class FCResNet(nn.Module):
    def __init__(
        self,
        #input_dim,
        features,
        #depth,
        spectral_normalization,
        coeff=5.0,
        coeff_fc=0.95,
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

        if spectral_normalization:
            # IMAGE
            self.first = spectral_norm_conv(
                self.first, 
                coeff=coeff, 
                input_dim=dims[0], 
                n_power_iterations=n_power_iterations
            )

            for i in range(len(self.res_second)):
                self.res_second[i] = spectral_norm_conv(
                    self.res_second[i],
                    coeff=coeff,
                    input_dim=dims[1] if i!=1 else dims[2],
                    n_power_iterations=n_power_iterations,
                )

            for i in range(len(self.res_third)):
                self.res_third[i] = spectral_norm_conv(
                    self.res_third[i],
                    coeff=coeff,
                    input_dim=dims[2] if i!=1 else dims[3],
                    n_power_iterations=n_power_iterations,
                )

            for i in range(len(self.res_fourth)):
                self.res_fourth[i] = spectral_norm_conv(
                    self.res_fourth[i],
                    coeff=coeff,
                    input_dim=dims[3] if i!=1 else dims[4],
                    n_power_iterations=n_power_iterations,
                )

            for i in range(len(self.res_fifth)):
                self.res_fifth[i] = spectral_norm_conv(
                    self.res_fifth[i],
                    coeff=coeff,
                    input_dim=dims[4] if i!=1 else dims[5],
                    n_power_iterations=n_power_iterations,
                )

            self.reduce = spectral_norm_fc(
                self.reduce,
                coeff=coeff_fc,
                n_power_iterations=n_power_iterations
            )

            # MEASUREMENTS
            self.first_m = spectral_norm_fc(
                self.first_m,
                coeff=coeff_fc,
                n_power_iterations=n_power_iterations
            )

            for i in range(len(self.measurement_encoder)):
                self.measurement_encoder[i] = spectral_norm_fc(
                    self.measurement_encoder[i],
                    coeff=coeff_fc,
                    n_power_iterations=n_power_iterations,
                )

        if activation == "relu":
            self.activation = F.relu
        elif activation == "elu":
            self.activation = F.elu
        else:
            raise ValueError("That acivation is unknown")

    def forward(self, front_img, state):
        # image encoder
        #print(front_img.shape)
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

def random_ortho(n, m):
    q, _ = torch.linalg.qr(torch.randn(n, m))
    return q

class RandomFourierFeatures(nn.Module):
    def __init__(self, in_dim, num_random_features, feature_scale=None):
        super().__init__()
        if feature_scale is None:
            feature_scale = math.sqrt(num_random_features / 2)

        self.register_buffer("feature_scale", torch.tensor(feature_scale))

        if num_random_features <= in_dim: # false
            W = random_ortho(in_dim, num_random_features) 
            # W shape (in_dim=16, num_random_features=32)
        else: # true
            # generate blocks of orthonormal rows which are not neccesarily orthonormal
            # to each other.
            dim_left = num_random_features # 32
            ws = []
            while dim_left > in_dim:
                ws.append(random_ortho(in_dim, in_dim)) # 16x16
                #print(random_ortho(in_dim, in_dim).shape)
                dim_left -= in_dim
            ws.append(random_ortho(in_dim, dim_left)) # 16x16
            #print(dim_left)
            #print(random_ortho(in_dim, dim_left).shape)
            W = torch.cat(ws, 1) # 16x32

        # From: https://github.com/google/edward2/blob/d672c93b179bfcc99dd52228492c53d38cf074ba/edward2/tensorflow/initializers.py#L807-L817
        feature_norm = torch.randn(W.shape) ** 2 # samples from N(0,1), 16x32
        W = W * feature_norm.sum(0).sqrt() # sum(0) reduces axis=0 -> 32 
        self.register_buffer("W", W) 
        # random feature matrix with orthogonal columns and gaussian like column norms
        # W has shape 16x32

        b = torch.empty(num_random_features).uniform_(0, 2 * math.pi) # 32
        self.register_buffer("b", b)

    def forward(self, x):
        k = torch.cos(x @ self.W + self.b)
        k = k / self.feature_scale

        return k

class DAVE2_SNGP(nn.Module):

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

        self.num_gp_features = num_gp_features
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
            self.normalize_steer = nn.LayerNorm(self.num_gp_features)
            self.normalize_accel = nn.LayerNorm(self.num_gp_features)

        self.rff_steer = RandomFourierFeatures(
            self.num_gp_features, num_random_features, feature_scale
        )
        self.rff_accel = RandomFourierFeatures(
            self.num_gp_features, num_random_features, feature_scale
        )

        self.beta_steer = nn.Linear(num_random_features, 1)
        self.beta_accel = nn.Linear(num_random_features, 2)

        self.num_data = num_data
        self.register_buffer("seen_data", torch.tensor(0))

        precision_steer = torch.eye(num_random_features) * self.ridge_penalty
        self.register_buffer("precision_steer", precision_steer)
        precision_accel = torch.eye(num_random_features) * self.ridge_penalty
        self.register_buffer("precision_accel", precision_accel)

        self.recompute_covariance = True
        self.register_buffer("covariance_steer", torch.eye(num_random_features))
        self.register_buffer("covariance_accel", torch.eye(num_random_features))

    def reset_precision_matrix(self):
        identity = torch.eye(self.precision_steer.shape[0], device=self.precision_steer.device)
        self.precision_steer = identity * self.ridge_penalty
        identity = torch.eye(self.precision_accel.shape[0], device=self.precision_accel.device)
        self.precision_accel = identity * self.ridge_penalty
        self.seen_data = torch.tensor(0)
        self.recompute_covariance = True

    def mean_field_logits(self, logits, pred_cov):
        # Mean-Field approximation as alternative to MC integration of Gaussian-Softmax
        # Based on: https://arxiv.org/abs/2006.07584

        logits_scale = torch.sqrt(1.0 + torch.diag(pred_cov) * self.mean_field_factor)
        if self.mean_field_factor > 0:
            logits = logits / logits_scale.unsqueeze(-1)

        return logits

    def forward(self, front_img, state=None):
        outputs = {}
        f, m = self.feature_extractor(front_img, state)
        #print(f.shape) # (128, (1024+256) 1280)
        outputs['feature_map'] = m
        outputs['feature_encoded'] = f
        f_reduc = self.jl(f)
        if self.normalize_gp_features:
            f_reduc = self.normalize_steer(f_reduc)
            f_reduc = self.normalize_accel(f_reduc)

        k_steer = self.rff_steer(f_reduc)
        outputs['projection_steer'] = k_steer
        pred_steer = self.beta_steer(k_steer)
        #pred_steer = self.tanh(pred_steer)
        pred_steer = torch.tanh(pred_steer)
        outputs['pred_steer'] = pred_steer 

        k_accel = self.rff_accel(f_reduc)
        outputs['projection_accel'] = k_accel
        pred_ab = self.beta_accel(k_accel)
        #pred_accel = self.tanh(pred_accel)
        pred_accel = (torch.tanh(pred_ab[:, 0])+1)/2
        pred_brake = (torch.tanh(pred_ab[:, 1])+1)/2
        outputs['pred_accel'] = pred_accel 
        outputs['pred_brake'] = pred_brake

        if self.training:
            precision_minibatch_steer = k_steer.t() @ k_steer
            precision_minibatch_accel = k_accel.t() @ k_accel
            self.precision_steer += precision_minibatch_steer
            self.precision_accel += precision_minibatch_accel
            self.seen_data += front_img.shape[0]

            assert (
                self.seen_data <= self.num_data
            ), "Did not reset precision matrix at start of epoch"
        else:
            assert self.seen_data > (
                self.num_data - self.train_batch_size
            ), "Not seen sufficient data for precision matrix"

            if self.recompute_covariance:
                with torch.no_grad():
                    eps = 1e-7
                    jitter = eps * torch.eye(
                        self.precision_steer.shape[1],
                        device=self.precision_steer.device,
                    )
                    # computes inverse of precision_steer/_accel -> posterior covariance (sigma)
                    u_steer, info_steer = torch.linalg.cholesky_ex(self.precision_steer + jitter)
                    u_accel, info_accel = torch.linalg.cholesky_ex(self.precision_accel + jitter)
                    assert (info_steer == 0).all(), "Precision matrix inversion failed!"
                    assert (info_accel == 0).all(), "Precision matrix inversion failed!"
                    torch.cholesky_inverse(u_steer, out=self.covariance_steer)
                    torch.cholesky_inverse(u_accel, out=self.covariance_accel)

                self.recompute_covariance = False

            with torch.no_grad():
                # compute posterior variance
                pred_cov_steer = k_steer @ ((self.covariance_steer @ k_steer.t()) * self.ridge_penalty)
                pred_cov_accel = k_accel @ ((self.covariance_accel @ k_accel.t()) * self.ridge_penalty)
                outputs['variance_steer'] = pred_cov_steer.diagonal().reshape(-1, 1) 
                outputs['variance_accel'] = pred_cov_accel.diagonal().reshape(-1, 1) 

        return outputs
    