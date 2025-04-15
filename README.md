# domain_aware_via_sngp_for_e2e_ad
Repository with additional information for paper "Domain Awareness via Spectral-normalized Neural Gaussian Processes for E2E Autonomous Vehicle Control".

## Abstract
The ability to quantify and understand uncertainty is crucial for improving the safety and reliability of autonomous vehicle systems. In this work, we introduce a novel domain awareness mechanism for end-to-end (E2E) autonomous driving algorithms by integrating Spectral-normalized Neural Gaussian Processes (SNGP) for deterministic uncertainty quantification into an E2E trainable autonomous driving framework. The goal is to enable the model trained on simulated data from CARLA to distinguish between unseen CARLA and real-world nuScenes scenarios during inference. Our results demonstrate that, after re-calibration, the model can effectively quantify the domain gap between simulated and real-world data. We found a 13% increase in throttle uncertainty when giving our model nuScenes instead of CARLA data. Additionally our experiments show that the quality of the predicted probability distributions is not influenced by the input domain. We further highlight, that the predictive ability of the E2E model is not affected by the network alterations introduced by SNGP.
![System overview](https://github.com/croth2305/domain_aware_via_sngp_for_e2e_ad/blob/main/pics/overview.png)

## Training with CARLA Data
Collect CARLA data using the CARLA AD Leaderboard in a running CARLA 0.9.10.1 instance by calling the data collection script. Make sure all paths in the script are configured correctly
```
sh code/DS/leaderboard/scripts/data_collection.sh
```


## Acknowledgments
SNGP Paper: https://arxiv.org/abs/2205.00403

SNGP Implementation: https://github.com/y0ast/DUE http://arxiv.org/abs/2102.11409

DAVE-2: http://arxiv.org/abs/1604.07316

Calibration: http://arxiv.org/abs/2207.01242

Pipeline from TCP: https://github.com/OpenDriveLab/TCP http://arxiv.org/abs/2206.08129

CARLA AD Leaderboard: http://leaderboard.carla.org/
