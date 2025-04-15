# domain_aware_via_sngp_for_e2e_ad
Repository with additional information for paper "Domain Awareness via Spectral-normalized Neural Gaussian Processes for E2E Autonomous Vehicle Control". :red_car:

## Abstract
The ability to quantify and understand uncertainty is crucial for improving the safety and reliability of autonomous vehicle systems. In this work, we introduce a novel domain awareness mechanism for end-to-end (E2E) autonomous driving algorithms by integrating Spectral-normalized Neural Gaussian Processes (SNGP) for deterministic uncertainty quantification into an E2E trainable autonomous driving framework. The goal is to enable the model trained on simulated data from CARLA to distinguish between unseen CARLA and real-world nuScenes scenarios during inference. Our results demonstrate that, after re-calibration, the model can effectively quantify the domain gap between simulated and real-world data. We found a 13% increase in throttle uncertainty when giving our model nuScenes instead of CARLA data. Additionally our experiments show that the quality of the predicted probability distributions is not influenced by the input domain. We further highlight, that the predictive ability of the E2E model is not affected by the network alterations introduced by SNGP.
![System overview](https://github.com/croth2305/domain_aware_via_sngp_for_e2e_ad/blob/main/pics/overview.png)

## Training and Evaluation
### Training with CARLA Data
Collect CARLA data using the CARLA AD Leaderboard in a running CARLA 0.9.10.1 instance by calling the data collection script. Make sure all paths in the script are configured correctly
```
sh code/DS/leaderboard/scripts/data_collection.sh
```
Run the scripts
```
sh code/DS/tools/filter_data.py
sh code/DS/tools/gen_data.py
```
using the paths specified during data collection.

For training, make sure you set your hyperparameters and data paths in ```code/DS/DS/config.py``` and then run the training script
```
sh code/DS/DS/train-gpu.py --id  "my_own_model" --batch_size 256 --logdir "path_to_my_log_dir" --gpus 1
```
The weights will be saved after every validation epoch as .pt files. :heart_eyes:

### Evaluation with Calibration
#### DATA PATHS
In ```calibration.py``` you need to set multiple paths:
- fitpaths: Paths to CARLA scenarios that will be used to fit the calibration GPs
- extrapaths_carla: Paths to CARLA Towns that will be used for evaluation, make sure to include Towns not seen during training
- extrapaths_nuscenes: Paths to nuScenes (or what ever real world data you want to use :sunglasses:) data that will be used for evaluation

#### EXTRACT DATA
Now you will first generate the mean and standard deviations for the image normalization:
- call the function ```get_mean_and_var_from_images(extrapaths)``` which you find somewhere around line 305 in ```carlibration.py```
- make sure to also call the ```exit(0)``` afterwards
Depending on the amount of CARLA data you specified, this might take very long... If necessary reduce the amount of CARLA data by removing Towns from extrapaths_carla :eyes:.
You'll get two .pt files in /DS/calibration. Make sure two comment out the function and the ```exit(0)``` again.

#### MODEL PATH
To ```path_to_conf_file``` add the path to the .pt file of the DAVE2_SNGP model you trained. Make sure the hyperparameters in ```config.py``` are the correct ones for your chosen model. Should you be using the DAVE2_Vanilla model, make sure that your path contains the word "vanilla".
Simply run the script with
 ```
python calibration.py
```
After running the script you should get multiple csv-files in the ```DS/calibration``` folder. :heart_eyes:

#### PLOT LOSSES :point_right: Visualization
You can use the notebook ```code/plot_losses.ipynb``` to process and visualize the data collected during training and evaluation. The scripts in this notebook were created during the research and might need to be updated to changed paths or data formats. Hope they still help! :innocent:

## Ideas
In case you want to continue this work, here are some topics you'd may address:
- Not using DAVE-2 but a more complex E2E algorithm
- Switching from RFF GP approximation to inducing point GPs, this might alliviate the need for the additional calibration ([DUE][link to due])
- Not only using nuScenes but also different real-world datasets
:sunglasses:

## Acknowledgments
- SNGP [Paper][link to sngp] and PyTorch [Implementation][link to due git]
- [DAVE-2 Model and Paper][link to dave]
- Calibration (GP-Normal) [Paper][link to cali] and [Implementation][link to cali git]
- Processing Pipeline taken from [TCP][link to tcp git]
- Data collection and evaluation using the [CARLA AD Leaderboard][link to carla]

:green_heart:

[link to due]:https://arxiv.org/abs/2102.11409
[link to sngp]:https://arxiv.org/abs/2205.00403
[link to due git]:https://github.com/y0ast/DUE
[link to dave]:http://arxiv.org/abs/1604.07316
[link to cali]:http://arxiv.org/abs/2207.01242
[link to cali git]:https://github.com/EFS-OpenSource/calibration-framework/tree/main
[link to tcp git]:https://github.com/OpenDriveLab/TCP
[link to carla]:http://leaderboard.carla.org/
