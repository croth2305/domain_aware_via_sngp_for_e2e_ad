import argparse
import os
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.distributions import Beta

from DS.model import DAVE2_SNGP,FCResNet
from DS.data import CARLA_Data
from DS.config import GlobalConfig

if __name__ == "__main__":
    # Config
    config = GlobalConfig()

    # Data
    val_set = CARLA_Data(root=config.root_dir_all, data_folders=[os.path.join(config.root_dir_all, 'town05' +'_short'),])
    dataloader_val = DataLoader(val_set, batch_size=None, shuffle=False, num_workers=1)
    
    feature_extractor = FCResNet(
        features=config.features,
        spectral_normalization=config.spectral_normalization,
        coeff=config.coeff,
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

    #path_to_conf_file = "log/DAVE2-SNGP-GPU-nospecnorm-1/best_model_60_loss=0.0048.pt"
    path_to_conf_file = "log/DAVE2-SNGP-GPU-c5/best_model_60_loss=0.0041.pt"

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

    train_loss = [] 

    i = 0
    fmap = []
    for batch in dataloader_val:
        model.eval()
        
        front_img = batch['front_img'].unsqueeze(0)
        speed = torch.Tensor([batch['speed'] / 12.])
        target_point = torch.Tensor([
            float(batch['target_point'][0]),
            float(batch['target_point'][1])])
        cmd = []
        for c in batch['target_command']:
            cmd.append(float(c))
        command = torch.Tensor(cmd)

        state = torch.cat([speed, target_point, command], 0).unsqueeze(0)

        #if i in range(10, 20):
            #front_img = torch.ones_like(front_img)*255
            #plt.figure(i)
            #plt.imshow(front_img[0, 0, :, :])
            #state = torch.zeros_like(state)
        
        y = torch.cat([
			batch['action'][1].reshape(-1, 1), 
			batch['action'][0].reshape(-1, 1), 
			batch['action'][2].reshape(-1, 1)], 1)

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

        fmap.append(y_pred['feature_map'].squeeze().cpu().detach().numpy())
 
        #if 7 < i < 12:
        #    for p in range(1, 65):
        #        plt.figure(i+1000, figsize=(20,20))
        #        plt.subplot(8, 8, p)
        #        plt.suptitle(f"var steer: {y_pred['variance_steer'].item()}, var accel: {y_pred['variance_accel'].item()}")
        #        plt.imshow(y_pred['feature_map'].cpu().detach().numpy()[0, p-1, :, :])
            #exit(0)

        #if i == 8:
        #    fm8 = y_pred['feature_map'].cpu().detach().numpy()
        #if i == 7:
        #    fm7 = y_pred['feature_map'].cpu().detach().numpy()
        #if i == 12:
        #    fm12 = y_pred['feature_map'].cpu().detach().numpy()
        #    for p in range(1, 65):
        #        plt.figure("7-12", figsize=(20,20))
        #        plt.subplot(8, 8, p)
        #        plt.imshow(fm7[0, p-1, :, :]-fm12[0, p-1, :, :])
        #    for p in range(1, 65):
        #        plt.figure("7-8", figsize=(20,20))
        #        plt.subplot(8, 8, p)
        #        plt.imshow(fm7[0, p-1, :, :]-fm8[0, p-1, :, :])

        steer_loss = loss_fn(y_pred['pred_steer'], y[0, 0].reshape(-1, 1))
        accel_loss = loss_fn(a.unsqueeze(dim=1), y[0, 1].reshape(-1, 1))
        brake_loss = loss_fn(b.unsqueeze(dim=1), y[0, 2].reshape(-1, 1))
        loss = steer_loss + accel_loss + brake_loss
        train_loss.append([
			str(steer_loss.item()), 
			str(accel_loss.item()), 
			str(brake_loss.item()), 
			str(loss.item()),
			str(y[0, 0].item()),
			str(y[0, 1].item()),
			str(y[0, 2].item()),
            str(y_pred['pred_steer'].item()),
            str(a.item()),
            str(b.item()),
            str(y_pred['variance_steer'].item()),
            str(y_pred['variance_accel'].item()),
            str(val_set.front_img[i]).split("/")[2]
            ])
        i += 1
        #exit(0)

    with open(
        os.path.join(os.path.join(
            'log', 
            'single-eval-c5'), 
                    "train_loss_array.txt"), "w") as txt_file:
        for line in train_loss:
            txt_file.write(" ".join(line) + "\n") 

    #exit(0)
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
    x = np.arange(len(train_loss[:100]))
    plt.figure(1)
    plt.plot(x, s_loss, c="b", label="steer loss")
    plt.plot(x, a_loss, c="r", label="accel loss")
    plt.plot(x, b_loss, c="g", label="brake loss")
    plt.legend()
    plt.figure(2)
    plt.plot(x, true_s, c="m", label="steer true")
    plt.plot(x, pred_s, c="k", label="steer pred")
    plt.legend()
    plt.figure(3)
    plt.plot(x, true_a, c="m", label="accel true")
    plt.plot(x, pred_a, c="k", label="accel pred")
    plt.legend()
    plt.figure(4)
    plt.plot(x, true_b, c="m", label="brake true")
    plt.plot(x, pred_b, c="k", label="brake pred")
    plt.legend()
    plt.figure(5)
    plt.plot(x, var_s, c="b", label="var steer")
    plt.plot(x, var_a, c="r", label="var acc")
    plt.legend() 

    #print(fmap[8:12])
    #x = np.arange(1280)
    #plt.figure(5)
    #plt.plot(x, fmap[0], c='b')
    #plt.plot(x, fmap[11], c='r')
    #plt.hlines(np.mean(fmap[0], axis=0), xmin=0, xmax=1280, colors='c')
    #plt.hlines(np.mean(fmap[11], axis=0), xmin=0, xmax=1280, colors='m')
    #plt.figure(6)
    #f1 = 0
    #f2 = 5
    #plt.scatter(fmap[:10][f1], fmap[:10][f2], c='b')
    #plt.scatter(fmap[20:][f1], fmap[20:][f2], c='b')
    #plt.scatter(fmap[10:20][f1], fmap[10:20][f2], c='r')
    #plt.figure(7)
    #f2 = 4
    #plt.scatter(fmap[:10][f1], fmap[:10][f2], c='b')
    #plt.scatter(fmap[20:][f1], fmap[20:][f2], c='b')
    #plt.scatter(fmap[10:20][f1], fmap[10:20][f2], c='r')
    plt.show()