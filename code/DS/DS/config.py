import os

class GlobalConfig:
	""" base architecture configurations """
	# Data
	seq_len = 1 # input timesteps
	pred_len = 4 # future waypoints predicted

	# data root
	root_dir_all = "data"

	train_towns = ['town02','town03', 'town04', 'town06']
	add_towns = ['town01', 'town02','town03', 'town04']
	l_towns = ['town01', 'town02']
	s_towns = ['town03', 'town02', 'town04']
	val_towns = ['town05']#['town02', 'town05', 'town07', 'town10']
	train_data, val_data = [], []
	for town in train_towns:		
		train_data.append(os.path.join(root_dir_all, town))
	for town in add_towns:
		train_data.append(os.path.join(root_dir_all, town+'_addition'))
	for town in l_towns:
		train_data.append(os.path.join(root_dir_all, town+'_long'))
	for town in s_towns:
		train_data.append(os.path.join(root_dir_all, town+'_short'))
		
	for town in val_towns:
		#val_data.append(os.path.join(root_dir_all, town+'_val'))
		val_data.append(os.path.join(root_dir_all, town+'_addition'))
		#val_data.append(os.path.join(root_dir_all, town+'_short'))

	ignore_sides = True # don't consider side cameras
	ignore_rear = True # don't consider rear cameras

	input_resolution = 256

	scale = 1 # image pre-processing
	crop = 256 # image pre-processing

	lr = 1e-4 # learning rate

	# Controller
	turn_KP = 0.75
	turn_KI = 0.75
	turn_KD = 0.3
	turn_n = 40 # buffer size

	speed_KP = 5.0
	speed_KI = 0.5
	speed_KD = 1.0
	speed_n = 40 # buffer size

	max_throttle = 0.75 # upper limit on throttle signal value in dataset
	brake_speed = 0.4 # desired speed below which brake is triggered
	brake_ratio = 1.1 # ratio of speed to desired speed at which brake is triggered
	clip_delta = 0.25 # maximum change in speed input to logitudinal controller


	aim_dist = 4.0 # distance to search around for aim point
	angle_thresh = 0.3 # outlier control detection angle
	dist_thresh = 10 # target point y-distance for outlier filtering


	speed_weight = 0.05
	value_weight = 0.001
	features_weight = 0.05

	rl_ckpt = "roach/log/ckpt_11833344.pth"

	img_aug = True

	# Feature Extractor
	features = 1024+256
	spectral_normalization = True
	#coeff = 0.95
	coeff = 5
	coeff_fc = 0.95
	n_power_iterations = 1
	dropout_rate = 0.01

	# Laplace and RFF 
	num_deep_features = 16
	num_gp_features = 16
	normalize_gp_features = True
	num_random_features = 32
	num_outputs = 1 # one output per GP -> two GPs
	num_data = 1000
	train_batch_size = 256
	ridge_penalty = 1.0 
	feature_scale = None,
	mean_field_factor = None


	def __init__(self, **kwargs):
		for k,v in kwargs.items():
			setattr(self, k, v)
