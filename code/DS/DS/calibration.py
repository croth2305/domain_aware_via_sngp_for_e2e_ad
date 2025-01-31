#https://github.com/EFS-OpenSource/calibration-framework?tab=readme-ov-file#probabilistic-regression

import numpy as np
import os
import json
import torch
import math
import pandas as pd
import seaborn as sns
from collections import OrderedDict
import matplotlib.pyplot as plt
from matplotlib.colorbar import ColorbarBase
import matplotlib.colors as colors
from PIL import Image
from torchvision import transforms as T
from netcal.metrics import NLL, PinballLoss, QCE
from netcal.presentation import ReliabilityRegression
from netcal.regression import VarianceScaling, GPNormal

import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.distributions import Beta
from torchvision import transforms as T

from DS.model import DAVE2_SNGP,FCResNet
from DS.vanilla import DAVE2_VANILLA
from DS.vanilla import FCResNet as FCResNetV
from DS.data import CARLA_Data
from DS.config import GlobalConfig

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 
def load_image(image):
    """
    Loads and resizes image to correct shape

    Parameters: image = path to image file
    Returns: (resized) image as torch.Tensor
    """
    #_im_transform = T.Compose([T.ToTensor(), T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])

    i = Image.open(image)
    if i.size != (900, 256):
        i = i.resize((900, 256))
    i = np.array(i)
    if len(i.shape) == 2:
        i = np.stack((i, i, i), axis=2)
    #resize = _im_transform(i)
    resize = i
    resize = torch.Tensor(resize).unsqueeze(0)
    try:
        resize = resize.reshape((1, 3, 256, 900))
    except:
        print(resize.shape, image)
    return resize

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 
def get_batches(dpath):
    """
    Finds needed information for prediction and ground truth for calibration

    Parameters: dpath = path to directory containing rgb, measurements, and supervision folder
    Returns: list with directory for each timestep in given scenario
    """
    # Data
    data = {} 
    speed = []
    target = []
    command = []
    action = []
    front = []
    for dir, folders, file in os.walk(dpath):
        for folder in folders:
            if folder=='rgb':
                p = os.path.join(dpath, folder)
                for _, _, f in sorted(os.walk(p)):
                    for item in f:
                        front.append(dpath+'/rgb/'+item)
            if folder=='measurements':
                p = os.path.join(dpath, folder)
                for _, _, f in sorted(os.walk(p)):
                    for m in f:
                        with open(os.path.join(p, m), "r") as read_file:
                            measurement = json.load(read_file)
                            speed.append(measurement['speed'])
                            target.append((measurement['x_target'], measurement['y_target']))
                            if 'target_command' in measurement.keys():
                                command.append(measurement['target_command'])
                            else:
                                command.append(measurement['command'])
                
            if folder=='supervision':
                p = os.path.join(dpath, folder)
                for _, _, f in sorted(os.walk(p)):
                    for m in f:
                        read_file = os.path.join(p, m)
                        supervision = np.load(read_file, allow_pickle=True).item()
                        action.append(supervision['action'])

    data['front_img'] = front
    data['action'] = action
    data['speed'] = speed
    data['target_point'] = target
    data['target_command'] = command

    b = []  
    for d in range(len(data["front_img"])):
        batch = {} 
        for k in data.keys():
            if k=='front_img':
                batch[k] = load_image(data[k][d])
                batch['name'] = data[k][d]
            else:
                batch[k] = data[k][d]
        b.append(batch)
    return b

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 
def get_prediction(batch, not_carla=False, mean=None, std=None):
    """
    Uses chosen model to predict steering, acceleration, and braking as well as lateral and longitudinal uncertainty

    Parameters: batch = directory containing data from one timestep
                not_carla = images that are not from CARLA will get normalized
    Returns:    ground truth data, 
                predicted steering and acceleration, 
                standard deviation for steering and acceleration,
                out_data for check if datasets are coherent
    """
    model.eval()
    
    front_img = batch['front_img']
    # normalize images that are not from carla
    if not_carla:
        # image = (image - mean) / std
        front_img = torch.div(torch.sub(front_img.reshape((256, 900, 3)).cuda(), mean.cuda()), std.cuda())
        front_img = front_img.reshape((1, 3, 256, 900))
        #front_img = front_img.squeeze(0)
        # https://github.com/google/uncertainty-baselines/blob/master/baselines/cifar/utils.py
    #print(front_img.shape)
    #front_img = front_img.reshape((1, 3, 256, 900))
    speed = torch.Tensor([batch['speed'] / 12.])
    target_point = torch.Tensor([
        float(batch['target_point'][0]),
        float(batch['target_point'][1])])
    c = batch['target_command']
    if c < 0:
        c = 4
    c -= 1
    assert c in [0, 1, 2, 3, 4, 5]
    cmd = [0] * 6
    cmd[c] = 1 
    command = torch.Tensor(cmd)

    state = torch.cat([speed, target_point, command], 0).unsqueeze(0)
    
    if type(batch['action'][1])==float:
        y = torch.cat([
            torch.Tensor(torch.Tensor([batch['action'][1]]).reshape(-1, 1)), 
            torch.Tensor(torch.Tensor([batch['action'][0]]).reshape(-1, 1)), 
            torch.Tensor(torch.Tensor([batch['action'][2]]).reshape(-1, 1))
            ], 1)
    else:
        y = torch.cat([
            torch.Tensor(batch['action'][1].reshape(-1, 1)), 
            torch.Tensor(batch['action'][0].reshape(-1, 1)), 
            torch.Tensor(batch['action'][2].reshape(-1, 1))
            ], 1)

    if torch.cuda.is_available():
        state = state.cuda()
        front_img = front_img.cuda()
        target_point = target_point.cuda()
        y = y.cuda()

    out_data ={ 
        "out_comma": batch['target_command'],
        "out_speed": batch['speed'],
        "out_steer": batch['action'][1] if type(batch['action'][1])==float else batch['action'][1].item(),
        "out_accel": batch['action'][0] if type(batch['action'][0])==float else batch['action'][0].item(),
        "out_brake": batch['action'][2] if type(batch['action'][2])==float else batch['action'][2].item(),
    }

    y_pred = model(front_img, state)
    a = y_pred['pred_accel']
    b = y_pred['pred_brake']
    for item in range(a.shape[-1]):
        if a[item] > b[item]:
            b[item] = 0
        else:
            a[item] = 0

    steer_loss = loss_fn(y_pred['pred_steer'], y[0, 0].reshape(-1, 1))
    accel_loss = loss_fn(a.unsqueeze(dim=1), y[0, 1].reshape(-1, 1))
            
    g = (batch['action'][1], batch['action'][0])
    m = (y_pred['pred_steer'].item(), a.item())
    s = (y_pred['variance_steer'].sqrt().item(), y_pred['variance_accel'].sqrt().item())
    return g, m, s, out_data, steer_loss.cpu().item(), accel_loss.cpu().item(), y_pred['feature_map'].item()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 
def get_means(std_g_throttle, std_throttle, std_g_steer, std_steer):
    """
    Calculates and prints mean values of standard deviations

    Parameters: std_g_throttle/steer = list of recalibrated stds for accel/steer
                std_throttle/steer = list of uncalibrated stds for accel/steer
    Returns: means of input parameters
    """
    thr_m_gp = np.mean(std_g_throttle.squeeze())
    thr_m_uc = torch.mean(std_throttle).item()
    ste_m_gp = np.mean(std_g_steer.squeeze())
    ste_m_uc = torch.mean(std_steer).item()
    if False:
        print("\n----------")
        print("Throttle")
        print("Mean stddev_gpnormal: ", thr_m_gp)
        print("Mean uncalibrated: ", thr_m_uc)
        print("Steer")
        print("Mean stddev_gpnormal: ", ste_m_gp)
        print("Mean uncalibrated: ", ste_m_uc)
        print("----------\n")
    return thr_m_gp, thr_m_uc, ste_m_gp, ste_m_uc

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 
def get_vars(std_g_throttle, std_throttle, std_g_steer, std_steer):
    """
    Calculates and prints variance values of standard deviations

    Parameters: std_g_throttle/steer = list of recalibrated stds for accel/steer
                std_throttle/steer = list of uncalibrated stds for accel/steer
    Returns: means of input parameters
    """
    thr_v_gp = np.var(std_g_throttle.squeeze())
    thr_v_uc = torch.var(std_throttle).item()
    ste_v_gp = np.var(std_g_steer.squeeze())
    ste_v_uc = torch.var(std_steer).item()
    return thr_v_gp, thr_v_uc, ste_v_gp, ste_v_uc

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 
def get_mean_and_var_from_images(paths):
    first = True
    for path in paths:
        batch = get_batches(path)
        for b in batch:
            img = torch.Tensor(b["front_img"]).cuda()
            if not first:
                channels = torch.cat([channels, img]).cuda()
            else:
                channels = img
                first = False

    # https://github.com/google/uncertainty-baselines/blob/master/baselines/cifar/utils.py
    # mean and std as one value for each channel
    mean = np.mean(channels.cpu().numpy(), axis=(0, 2, 3))
    std = np.std(channels.cpu().numpy(), axis=(0, 2, 3))
    means = torch.Tensor(mean)
    stds = torch.Tensor(std)

    if torch.isnan(means).any() or torch.isinf(means).any():
        print("nan or inf in means")
    if torch.isnan(stds).any() or torch.isinf(stds).any():
        print("nan or inf in stds")
    torch.save(means, "../DS/calibration/carla_img_mean.pt")
    torch.save(stds, "../DS/calibration/carla_img_std.pt")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 
if __name__ == "__main__":
    torch.cuda.empty_cache()
    config = GlobalConfig()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # DATA PATHS

    fitpaths = [
        "../DS/data/town05_short/routes_town05_short_01_04_12_17_28",
        "../DS/data/town05_short/routes_town05_short_01_04_12_19_50",
        "../DS/data/town05_short/routes_town05_short_01_04_12_20_26",
        "../DS/data/town05_short/routes_town05_short_01_04_12_25_40",
        "../DS/data/town05_short/routes_town05_short_01_04_12_27_25",
        "../DS/data/town05_short/routes_town05_short_01_04_12_27_58",
        "../DS/data/town05_short/routes_town05_short_01_04_12_30_21"
        ]

    extrapaths_carla = [
        "../DS/data/town05_short",
        "../DS/data/town02_short",
        "../DS/data/town02_long",
        "../DS/data/town03",
        "../DS/data/town04",
        "../DS/data/town01-tiny-weather",
        "../DS/data/town01-tiny-fov140",
        ]
    
    extrapaths_nuscenes = [
        "/home/carlas/Experiments/NuScenesE2E/boston-seaport/day",
        "/home/carlas/Experiments/NuScenesE2E/boston-seaport/day_rain",
        #"/home/carlas/Experiments/NuScenesE2E/singapore-hollandvillage/day",
        #"/home/carlas/Experiments/NuScenesE2E/singapore-hollandvillage/night",
        #"/home/carlas/Experiments/NuScenesE2E/singapore-hollandvillage/rain_night",
        #"/home/carlas/Experiments/NuScenesE2E/singapore-onenorth/day",
        #"/home/carlas/Experiments/NuScenesE2E/singapore-queenstown/day",
        #"/home/carlas/Experiments/NuScenesE2E/singapore-queenstown/night",
    ]

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # EXTRACT DATA

    extrapaths = []
    for path_carla in extrapaths_carla:
        carla = sorted(os.listdir(path_carla))
        for i in range(len(carla)):
            p = os.path.join(path_carla, carla[i])
            if os.path.isdir(p):
                extrapaths.append(p)

    #get_mean_and_var_from_images(extrapaths)
    #exit(0)
    img_mean = torch.load("../DS/calibration/carla_img_mean.pt")
    img_std = torch.load("../DS/calibration/carla_img_std.pt")
    #print(img_mean, img_std)
    #exit(0)

    for path_nuscenes in extrapaths_nuscenes:
        nuscenes = sorted(os.listdir(path_nuscenes))
        for i in range(len(nuscenes)):
            extrapaths.append(os.path.join(path_nuscenes, nuscenes[i]))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # MODEL PATH

    #path_to_conf_file = "log/DAVE2-SNGP-GPU-10/best_model_60_loss=0.0053.pt"
    #path_to_conf_file = "log/DAVE2-SNGP-GPU-nospecnorm-1/best_model_60_loss=0.0048.pt"
    #path_to_conf_file = "log/DAVE2-SNGP-GPU-c5/best_model_60_loss=0.0041.pt"
    #path_to_conf_file = "log/DAVE2-SNGP-GPU-c5and095/best_model_60_loss=0.0054.pt"
    #path_to_conf_file = "log/DAVE2-SNGP-GPU-vanilla/best_model_60_loss=0.0078.pt"
    #path_to_conf_file = "log/DAVE2-SNGP-GPU-noCommand/best_model_60_loss=0.0051.pt"
    #path_to_conf_file = "log/DAVE2-SNGP-GPU-miniData/best_model_60_loss=0.1357.pt"
    path_to_conf_file = "log/DAVE2-SNGP-GPU-c5and095-noNorm/best_model_60_loss=0.0091.pt"
    #path_to_conf_file = "log/DAVE2-SNGP-GPU-vanilla_noNorm/best_model_60_loss=0.0117.pt"
    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # MODEL

    feature_extractor = FCResNet(
        features=config.features,
        spectral_normalization=config.spectral_normalization,
        coeff=config.coeff,
        coeff_fc=config.coeff_fc, #!!!!!!!!!
        n_power_iterations=config.n_power_iterations,
        dropout_rate=config.dropout_rate
    )
    model = DAVE2_SNGP(
        feature_extractor=feature_extractor,
        num_deep_features=config.features,
        num_gp_features=config.num_gp_features,
        normalize_gp_features=config.normalize_gp_features,
        num_random_features=config.num_random_features,
        num_outputs=config.num_outputs,
        num_data=1,
        train_batch_size=1,
        ridge_penalty=1.0,
        feature_scale=None,
        mean_field_factor=None
    )
    loss_fn = F.mse_loss

    if "vanilla" in path_to_conf_file:
        print("vanilla")
        feature_extractor = FCResNetV(
            features=config.features,
            spectral_normalization=config.spectral_normalization,
            coeff=config.coeff,
            n_power_iterations=config.n_power_iterations,
            dropout_rate=config.dropout_rate
        )
        model = DAVE2_VANILLA(
            feature_extractor=feature_extractor,
            num_deep_features=config.features,
            num_gp_features=config.num_gp_features,
            normalize_gp_features=config.normalize_gp_features,
            num_random_features=config.num_random_features,
            num_outputs=config.num_outputs,
            num_data=1,
            train_batch_size=1,
            ridge_penalty=1.0,
            feature_scale=None,
            mean_field_factor=None
        )

    try:
        ckpt = torch.load(path_to_conf_file)
        ckpt = ckpt["state_dict"]
        new_state_dict = OrderedDict()
        for key, value in ckpt.items():
            new_key = key.replace("model.","")
            new_state_dict[new_key] = value
        model.load_state_dict(new_state_dict, strict = False)
    except:
        try:
            checkpoint = torch.load(path_to_conf_file)
            model.load_state_dict(checkpoint)
        except Exception as e:
            print(e)

    if torch.cuda.is_available():
        model = model.cuda()


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # FIT GP-NORMAL
    gt = []
    mean = []
    std = []
    for path in fitpaths:
        b = get_batches(path)
        for batch in b:
            g, m, s, _, _, _, _ = get_prediction(batch=batch)
            gt.append(g)
            mean.append(m)
            std.append(s)

    gt = np.asarray(gt)[:, :]
    gt_steer = gt[:, 0]
    gt_throttle = gt[:, 1]
    mean = torch.as_tensor(np.asarray(mean))
    mean_steer = mean[:, 0]
    mean_throttle = mean[:, 1]
    std = torch.as_tensor(np.asarray(std))
    std_steer = std[:, 0]
    std_throttle = std[:, 1]

    ground_truth = pd.DataFrame()
    ground_truth["ground truth steer"] = gt_steer
    ground_truth["ground truth throttle"] = gt_throttle
    ground_truth["predicted steer"] = mean_steer
    ground_truth["predicted throttle"] = mean_throttle
    ground_truth["predicted std steer"] = std_steer
    ground_truth["predicted std throttle"] = std_throttle
    #ground_truth.to_csv(path_or_buf="../DS/calibration/uncalibrated.csv", index=False)

    # ValueError: operands could not be broadcast together with shapes (296,691209) (296,2)
    # -> mean and std need dimx1 -> split into steer and throttle

    # PROBABILISTIC REGRESSION

    cali_steer = GPNormal(
        n_inducing_points=12,    # number of inducing points
        n_random_samples=256,    # random samples used for likelihood
        n_epochs=256,            # optimization epochs
        use_cuda=True,          # can also use CUDA for computations
    )
    cali_throttle = GPNormal(
        n_inducing_points=12,    # number of inducing points
        n_random_samples=256,    # random samples used for likelihood
        n_epochs=256,            # optimization epochs
        use_cuda=True,          # can also use CUDA for computations
    )
    cali_steer.fit((mean_steer, std_steer), gt_steer)
    cali_throttle.fit((mean_throttle, std_throttle), gt_throttle)

    everything = pd.DataFrame(columns=[
        'name', 
        'label', 
        'throttle_mean_gp', 
        'throttle_mean_uncali',
        'steer_mean_gp', 
        'steer_mean_uncali',
        ])
    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # data to compare scenario occurences in different datasets
   
    c_distributions = pd.DataFrame(columns=[
        'comma',
        'speed',
        'steer',
        'accel',
        'brake',
        ])
    n_distributions = pd.DataFrame(columns=[
        'comma',
        'speed',
        'steer',
        'accel',
        'brake',
        ])
 
    c_comma = []
    c_speed = []
    c_steer = []
    c_accel = []
    c_brake = []
    n_comma = []
    n_speed = []
    n_steer = []
    n_accel = []
    n_brake = []
    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # data for calibration plot

    plot_c_gt_steer = []
    plot_n_gt_steer = []
    plot_c_gt_throttle = []
    plot_n_gt_throttle = []

    plot_c_mean_steer = []
    plot_n_mean_steer = []
    plot_c_mean_throttle = []
    plot_n_mean_throttle = []

    plot_c_std_uncali_steer = []
    plot_n_std_uncali_steer = []
    plot_c_std_uncali_throttle = []
    plot_n_std_uncali_throttle = []

    plot_c_std_recali_steer = []
    plot_n_std_recali_steer = []
    plot_c_std_recali_throttle = []
    plot_n_std_recali_throttle = []

    plot_c_loss_steer = []
    plot_c_loss_throttle = []
    plot_n_loss_steer = []
    plot_n_loss_throttle = []

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # RECALIBRATE

    for path in extrapaths:
        try:
        #if True:
            b = get_batches(path)
            gt = []
            mean = []
            std = []
            loss_s = []
            loss_a = []
            for batch in b:
                if batch['target_command'] in [0, 1, 2, 3, 4, 5]:
                    if "NuScenes" in str(batch['name']):
                        g, m, s, out, l_s, l_a = get_prediction(batch=batch, not_carla=True, mean=img_mean, std=img_std)
                        #g, m, s, out = get_prediction(batch=batch)
                        label = 'nuscenes'
                        n_comma.append(out["out_comma"])
                        n_speed.append(out["out_speed"])
                        n_steer.append(out["out_steer"])
                        n_accel.append(out["out_accel"])
                        n_brake.append(out["out_brake"])
                    else:
                        g, m, s, out, l_s, l_a = get_prediction(batch=batch)
                        label = 'carla'
                        c_comma.append(out["out_comma"])
                        c_speed.append(out["out_speed"])
                        c_steer.append(out["out_steer"])
                        c_accel.append(out["out_accel"])
                        c_brake.append(out["out_brake"])
                    gt.append(g)
                    mean.append(m)
                    std.append(s)
                    loss_s.append(l_s)
                    loss_a.append(l_a)
                    name = path.split("/")[-1]

            gt = np.asarray(gt)[:, :]
            gt_steer = gt[:, 0]
            gt_throttle = gt[:, 1]
            mean = torch.as_tensor(np.asarray(mean))
            mean_steer = mean[:, 0]
            mean_throttle = mean[:, 1]
            std = torch.as_tensor(np.asarray(std))
            std_steer = std[:, 0]
            std_throttle = std[:, 1]
            loss_s = np.asarray(loss_s)
            loss_a = np.asarray(loss_a)

        except:
            print(path)

        #print(path)
        #print(std)

        std_g_steer = cali_steer.transform((mean_steer, std_steer))
        std_g_throttle = cali_throttle.transform((mean_throttle, std_throttle))

        nll = NLL()
        nll_un_steer = nll.measure((mean_steer, std_steer), gt_steer, reduction='mean')
        nll_re_steer = nll.measure((mean_steer, std_g_steer), gt_steer, reduction='mean')
        nll_un_throttle = nll.measure((mean_throttle, std_throttle), gt_throttle, reduction='mean')
        nll_re_throttle = nll.measure((mean_throttle, std_g_throttle), gt_throttle, reduction='mean')

        #print(name)
        #print("Throttle")
        #print("NLL stddev_gpnormal: ", nll_re_throttle)
        #print("NLL uncalibrated: ", nll_un_throttle)
        #print("Steer")
        #print("NLL stddev_gpnormal: ", nll_re_steer)
        #print("NLL uncalibrated: ", nll_un_steer)

        thr_m_gp, thr_m_uc, ste_m_gp, ste_m_uc = get_means(std_g_throttle, std_throttle, std_g_steer, std_steer)
        thr_v_gp, thr_v_uc, ste_v_gp, ste_v_uc = get_vars(std_g_throttle, std_throttle, std_g_steer, std_steer)

        n = name.split("_")[-1]
        if len(n)==2:
            n = 'carla'
        elif label == 'nuscenes':
            n = 'nuscenes'
            if "day" in str(batch['name']):
                n += " day"
            elif "night" in str(batch['name']):
                n += " night"
            if "rain" in str(batch['name']):
                n += " rain"


        new = pd.DataFrame({
            'name': [n], 
            'label': [label], 
            'throttle_mean_gp': [thr_m_gp], 
            'throttle_mean_uncali': [thr_m_uc],
            'steer_mean_gp': [ste_m_gp], 
            'steer_mean_uncali': [ste_m_uc],
            'throttle_nll_gp': [nll_re_throttle], 
            'throttle_nll_uncali': [nll_un_throttle],
            'steer_nll_gp': [nll_re_steer], 
            'steer_nll_uncali': [nll_un_steer],
            'throttle_var_gp': [thr_v_gp], 
            'throttle_var_uncali': [thr_v_uc],
            'steer_var_gp': [ste_v_gp], 
            'steer_var_uncali': [ste_v_uc],
            })
        everything = pd.concat([everything, new], ignore_index=True)

        if label == "carla" and len(loss_s) > 0:
            for e in range(len(gt_steer)):
                plot_c_gt_steer.append(gt_steer[e])
                plot_c_gt_throttle.append(gt_throttle[e])

                plot_c_mean_steer.append(mean_steer[e].item())
                plot_c_mean_throttle.append(mean_throttle[e].item())

                plot_c_std_uncali_steer.append(std_steer[e].item())
                plot_c_std_uncali_throttle.append(std_throttle[e].item())

                plot_c_std_recali_steer.append(std_g_steer[e][0])
                plot_c_std_recali_throttle.append(std_g_throttle[e][0])
                
                plot_c_loss_steer.append(loss_s[e])
                plot_c_loss_throttle.append(loss_a[e])
        elif len(loss_s) > 0:
            for e in range(len(gt_steer)):
                plot_n_gt_steer.append(gt_steer[e])
                plot_n_gt_throttle.append(gt_throttle[e])

                plot_n_mean_steer.append(mean_steer[e].item())
                plot_n_mean_throttle.append(mean_throttle[e].item())

                plot_n_std_uncali_steer.append(std_steer[e].item())
                plot_n_std_uncali_throttle.append(std_throttle[e].item())

                plot_n_std_recali_steer.append(std_g_steer[e][0])
                plot_n_std_recali_throttle.append(std_g_throttle[e][0])

                plot_n_loss_steer.append(loss_s[e])
                plot_n_loss_throttle.append(loss_a[e])

        #if "scene-0062" in name:
        #    cmap = sns.color_palette("rocket", as_cmap=True)
        #    mycolors = cmap(np.linspace(0, 1, 5))
        #    x_lin = np.linspace(0, mean_steer.shape[0], mean_steer.shape[0])
        #    plt.figure(name)
        #    plt.subplot(2, 1, 1)
        #    plt.ylim(-1.0, 1.0)
        #    plt.fill_between(x_lin, mean_steer - std_g_steer.squeeze(), mean_steer + std_g_steer.squeeze(), alpha=0.1, color=mycolors[2])
        #    plt.plot(x_lin, mean_steer, color=mycolors[2])
        #    plt.plot(x_lin, gt_steer, color="c")
        #    plt.subplot(2, 1, 2)
        #    plt.ylim(0.0, 1.0)
        #    plt.fill_between(x_lin, mean_throttle - std_g_throttle.squeeze(), mean_throttle + std_g_throttle.squeeze(), alpha=0.1, color=mycolors[2])
        #    plt.plot(x_lin, gt_throttle, color="c")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # SAVE DATA

    everything.to_csv(path_or_buf="../DS/calibration/DS-c5-095_noNorm.csv", index=False)

    c_distributions['comma'] = c_comma 
    c_distributions['speed'] = c_speed 
    c_distributions['steer'] = c_steer 
    c_distributions['accel'] = c_accel 
    c_distributions['brake'] = c_brake 
    n_distributions['comma'] = n_comma 
    n_distributions['speed'] = n_speed 
    n_distributions['steer'] = n_steer 
    n_distributions['accel'] = n_accel 
    n_distributions['brake'] = n_brake 
    c_distributions.to_csv(path_or_buf="../DS/calibration/c_scenario_dist_noNorm.csv", index=False)
    n_distributions.to_csv(path_or_buf="../DS/calibration/n_scenario_dist_noNorm.csv", index=False)

    plot_calibration_c = pd.DataFrame()
    plot_calibration_n = pd.DataFrame()
    plot_calibration_c["plot_c_gt_steer"] = plot_c_gt_steer
    plot_calibration_n["plot_n_gt_steer"] = plot_n_gt_steer
    plot_calibration_c["plot_c_gt_throttle"] = plot_c_gt_throttle
    plot_calibration_n["plot_n_gt_throttle"] = plot_n_gt_throttle
    plot_calibration_c["plot_c_mean_steer"] = plot_c_mean_steer
    plot_calibration_n["plot_n_mean_steer"] = plot_n_mean_steer
    plot_calibration_c["plot_c_mean_throttle"] = plot_c_mean_throttle
    plot_calibration_n["plot_n_mean_throttle"] = plot_n_mean_throttle
    plot_calibration_c["plot_c_std_uncali_steer"] = plot_c_std_uncali_steer
    plot_calibration_n["plot_n_std_uncali_steer"] = plot_n_std_uncali_steer
    plot_calibration_c["plot_c_std_uncali_throttle"] = plot_c_std_uncali_throttle
    plot_calibration_n["plot_n_std_uncali_throttle"] = plot_n_std_uncali_throttle
    plot_calibration_c["plot_c_std_recali_steer"] = plot_c_std_recali_steer
    plot_calibration_n["plot_n_std_recali_steer"] = plot_n_std_recali_steer
    plot_calibration_c["plot_c_std_recali_throttle"] = plot_c_std_recali_throttle
    plot_calibration_n["plot_n_std_recali_throttle"] = plot_n_std_recali_throttle

    plot_calibration_c["plot_c_loss_steer"] = plot_c_loss_steer
    plot_calibration_n["plot_n_loss_steer"] = plot_n_loss_steer
    plot_calibration_c["plot_c_loss_throttle"] = plot_c_loss_throttle
    plot_calibration_n["plot_n_loss_throttle"] = plot_n_loss_throttle
    plot_calibration_c.to_csv(path_or_buf="../DS/calibration/plot_calibration_c.csv", index=False)
    plot_calibration_n.to_csv(path_or_buf="../DS/calibration/plot_calibration_n.csv", index=False)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # CALIBRATION PLOT

    #quantiles = np.linspace(0.1, 0.9, 9)
    #diagram = ReliabilityRegression(quantiles=quantiles)
    #diagram.plot((mean_steer, std_steer), gt_steer, title_suffix=f"steer {n} uncalibrated")
    #diagram.plot((mean_steer, std_g_steer), gt_steer, title_suffix=f"steer {n} recalibrated")
    #diagram.plot((mean_throttle, std_throttle), gt_throttle, title_suffix=f"throttle {n} uncalibrated")
    #diagram.plot((mean_throttle, std_g_throttle), gt_throttle, title_suffix=f"throttle {n} recalibrated")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    plt.show()
