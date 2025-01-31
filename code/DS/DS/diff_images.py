import argparse
import os
from collections import OrderedDict
import matplotlib.pyplot as plt
from matplotlib.colorbar import ColorbarBase
import matplotlib.colors as colors
import numpy as np
import json
from PIL import Image
import cv2

import torch
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

if __name__ == "__main__":
    # Config
    config = GlobalConfig()

    # Data
    data = {} 
    speed = []
    target = []
    command = []
    action = []
    front = []
    dpaths = [
        # data paths to scenarios from CARLA or different domain
        # path needs to go to folder that contains "rgb", "measurements" and "supervision"
        "../DS/data/town05_short/routes_town05_short_01_04_12_30_21"
        ] 
    for dpath in dpaths:
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
                                command.append(measurement['target_command'])
                    
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
    
    feature_extractor = FCResNet(
        features=config.features,
        spectral_normalization=config.spectral_normalization,
        coeff=config.coeff,
        coeff_fc=config.coeff_fc,
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

    #path_to_conf_file = "log/DAVE2-SNGP-GPU-10/best_model_60_loss=0.0053.pt"
    #path_to_conf_file = "log/DAVE2-SNGP-GPU-nospecnorm-1/best_model_60_loss=0.0048.pt"
    #path_to_conf_file = "log/DAVE2-SNGP-GPU-c5/best_model_60_loss=0.0041.pt"
    #path_to_conf_file = "log/DAVE2-SNGP-GPU-c5and095/best_model_60_loss=0.0054.pt"
    #path_to_conf_file = "log/DAVE2-SNGP-GPU-vanilla/best_model_60_loss=0.0078.pt"
    #path_to_conf_file = "log/DAVE2-SNGP-GPU-noCommand/best_model_60_loss=0.0051.pt"
    #path_to_conf_file = "log/DAVE2-SNGP-GPU-miniData/best_model_60_loss=0.1357.pt"
    #path_to_conf_file = "log/DAVE2-SNGP-GPU-c5and095-noNorm/best_model_60_loss=0.0091.pt"
    path_to_conf_file = "log/DAVE2-SNGP-GPU-vanilla_noNorm/best_model_60_loss=0.0117.pt"
    
    if "vanilla" in path_to_conf_file:
        feature_extractor = FCResNetV(
            features=config.features,
            spectral_normalization=False,
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

    # TODO: parameters of full vs vanilla model
    param_size = 0
    params = 0
    for param in model.parameters():
        params += param.nelement()
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.3f}MB'.format(size_all_mb))
    print('parameters: {}'.format(params))

    # TODO: runtime comparison full vs vanilla
    # https://towardsdatascience.com/the-correct-way-to-measure-inference-time-of-deep-neural-networks-304a54e5187f

    # TODO: calculate MSE of full vs vanilla model for one scenario
    train_loss = [] 

    i = 0
    fmap = []
    fmap_max = 0

    def load_image(image):
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
    
    b = []  
    for d in range(len(data["front_img"])):
        batch = {} 
        for k in data.keys():
            if k=='front_img':
                batch[k] = load_image(data[k][d])
            else:
                batch[k] = data[k][d]
        b.append(batch)
    #print(b)

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    r = len(b)
    timings = np.zeros((r, 1))
    mse_s = np.zeros((r, 1))
    mse_a = np.zeros((r, 1))
    rep = 0
    
    for batch in b:
        starter.record()
        model.eval()
        
        front_img = batch['front_img']
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

        y_pred = model(front_img, state)
        a = y_pred['pred_accel']
        b = y_pred['pred_brake']
        for item in range(a.shape[-1]):
            if a[item] > b[item]:
                b[item] = 0
            else:
                a[item] = 0

        ender.record()
        # WAIT FOR GPU SYNC
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        timings[rep] = curr_time
        #map = y_pred['feature_map'].squeeze().cpu().detach().numpy()
        #print(y_pred['projection_steer'].squeeze().cpu().detach().numpy().shape)
        #print(y_pred['projection_accel'].squeeze().cpu().detach().numpy().shape)
        #fmap.append(map)
        #fmap_maximum = np.max(map)
        #if fmap_maximum > fmap_max:
            #fmap_max = fmap_maximum
 
        steer_loss = loss_fn(y_pred['pred_steer'], y[0, 0].reshape(-1, 1))
        accel_loss = loss_fn(a.unsqueeze(dim=1), y[0, 1].reshape(-1, 1))
        brake_loss = loss_fn(b.unsqueeze(dim=1), y[0, 2].reshape(-1, 1))
        loss = steer_loss + accel_loss + brake_loss

        mse_s[rep] = steer_loss.cpu().item()
        mse_a[rep] = accel_loss.cpu().item()

        rep += 1
    
        #train_loss.append([
		#	str(steer_loss.item()), 
		#	str(accel_loss.item()), 
		#	str(brake_loss.item()), 
		#	str(loss.item()),
		#	str(y[0, 0].item()),
		#	str(y[0, 1].item()),
		#	str(y[0, 2].item()),
        #    str(y_pred['pred_steer'].item()),
        #    str(a.item()),
        #    str(b.item()),
        #    str(y_pred['variance_steer'].item()) if type(model) == DAVE2_SNGP else str(0),
        #    str(y_pred['variance_accel'].item()) if type(model) == DAVE2_SNGP else str(0),
        #    str(data['front_img'][i])#.split("/")[2]
        #    ])
        #i += 1

    mean_syn = np.sum(timings)/r
    std_syn = np.std(timings)

    mean_ms = np.sum(mse_s)/r
    std_ms = np.std(mse_s)

    mean_ma = np.sum(mse_a)/r
    std_ma = np.std(mse_a)

    print("############################")
    print("Model: \t", path_to_conf_file)
    print("Runtime: \t", round(mean_syn, 4), " +-", round(std_syn, 4))
    print("MSE steer: \t", round(mean_ms, 4), " +-", round(std_ms, 4))
    print("MSE accel: \t", round(mean_ma, 4), " +-", round(std_ma, 4))
    print("############################")

    exit(0)

    with open(
        os.path.join(os.path.join(
            'log', 
            'single-eval-vanilla-vs-full'), 
                    "train_loss_array-fullnospec.txt"), "w") as txt_file:
        for line in train_loss:
            txt_file.write(" ".join(line) + "\n") 

    exit(0)
    s_loss = []
    a_loss = []
    b_loss = []
    loss = []
    true_s = []
    true_a = []
    true_b = []
    pred_s = []
    pred_a = []
    pred_b = []
    var_s = []
    var_a = []
    var = []
    img = [] 
    for res in train_loss[:100]:
        s_loss.append(float(res[0]))
        a_loss.append(float(res[1]))
        b_loss.append(float(res[2]))
        loss.append(float(res[3]))
        true_s.append(float(res[4]))
        true_a.append(float(res[5]))
        true_b.append(float(res[6]))
        pred_s.append(float(res[7]))
        pred_a.append(float(res[8]))
        pred_b.append(float(res[9]))
        var_s.append(float(res[10]))
        var_a.append(float(res[11]))
        var.append(float(res[10])+float(res[11]))
        img.append(res[12])

    x = np.arange(len(train_loss[:100]))
    
    def where_switch(folder:str):
        switch = 0
        for im in img:
            #print(im)
            if folder in str(im):
                return float(switch)
            switch += 1
        return float(switch)
    
    switch = [] 
    for path in dpaths:
        switch.append(where_switch(path))

    plt.figure(1)
    plt.vlines(switch, 0, 1, colors="c")
    plt.plot(x, s_loss, c="b", label="steer loss")
    plt.plot(x, a_loss, c="r", label="accel loss")
    plt.plot(x, b_loss, c="g", label="brake loss")
    plt.legend()
    plt.figure(2)
    plt.vlines(switch, 0, 1, colors="c")
    plt.plot(x, true_s, c="m", label="steer true")
    plt.plot(x, pred_s, c="k", label="steer pred")
    plt.legend()
    plt.figure(3)
    plt.vlines(switch, 0, 1, colors="c")
    plt.plot(x, true_a, c="m", label="accel true")
    plt.plot(x, pred_a, c="k", label="accel pred")
    plt.legend()
    plt.figure(4)
    plt.vlines(switch, 0, 1, colors="c")
    plt.plot(x, true_b, c="m", label="brake true")
    plt.plot(x, pred_b, c="k", label="brake pred")
    plt.legend()
    plt.figure(5)
    top = np.max(var)+0.01
    plt.vlines(switch, -0.0002, top, colors="c")
    #print(switch)
    for s in range(len(switch)):
        if s != len(switch)-1:
            mean = str(
                round(
                    np.mean(
                        var[int(switch[s]):int(switch[s+1])]), 6))
            
            means = str(
                round(
                    np.mean(
                        var_s[int(switch[s]):int(switch[s+1])]), 6))
            
            meana = str(
                round(
                    np.mean(
                        var_a[int(switch[s]):int(switch[s+1])]), 6))
        else:
            mean = str(
                round(
                    np.mean(
                        var[int(switch[s]):]), 6))
            means = str(
                round(
                    np.mean(
                        var_s[int(switch[s]):]), 6))
            meana = str(
                round(
                    np.mean(
                        var_a[int(switch[s]):]), 6))
        plt.annotate(f"{mean}\n{means}\n{meana}\n", (switch[s], 0))
        #plt.annotate(means, (switch[s], -0.00006), color="b")
        #plt.annotate(meana, (switch[s], -0.00012), color="r")
        path = dpaths[s].split("/")[-1]
        if "_" in path:
            path = path.split("_")[-1]
        plt.annotate(path, (switch[s], top-0.0005))
    plt.plot(x, var_s, c="b", label="var steer")
    plt.plot(x, var_a, c="r", label="var acc")
    plt.plot(x, var, c="k", label="var")
    plt.legend() 

    plt.figure()
    l = len(switch)
    #l = len(fmap)
    cmap = plt.cm.viridis
    vmin=0
    vmax=fmap_max
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    for im in range(l):
            #plt.figure()
            plt.subplot(2, int(l/2), im+1)
            path = dpaths[im].split("/")[-1]
            if "_" in path:
                path = path.split("_")[-1]
            plt.title(path)
            #print(img[int(switch[im])])
            img1 = plt.imshow(fmap[int(switch[im])].sum(0), cmap=cmap, norm=norm)
            #img1 = plt.imshow(fmap[int(im)].sum(0), cmap=cmap, norm=norm)
            #print(os.path.split(img[im])[0])
            #dir, file = os.path.split(img[im])
            #plt.imsave(dir + "-featuremaps/" + file, fmap[int(im)].sum(0), vmin=vmin, vmax=vmax, cmap=cmap, dpi=10000)
            #plt.close()
    #cbar = fig.colorbar(img1, ax=axs)
    #cbar.set_label('Activation')

    plt.show()
