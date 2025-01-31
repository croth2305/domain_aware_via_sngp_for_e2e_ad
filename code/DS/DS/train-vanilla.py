import argparse
import os
from collections import OrderedDict

import cv2

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.distributions import Beta

from ignite.engine import Events, Engine
from ignite.metrics import Average, Loss
from ignite.contrib.handlers import ProgressBar
from ignite.handlers import Checkpoint, global_step_from_engine

from DS.vanilla import DAVE2_VANILLA, FCResNet
from DS.data import CARLA_Data
from DS.config import GlobalConfig

if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument('--id', type=str, default='DAVE2-SNGP-GPU-vanilla_noNorm', help='Unique experiment identifier.')
	parser.add_argument('--epochs', type=int, default=60, help='Number of train epochs.')
	parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate.')
	parser.add_argument('--val_every', type=int, default=3, help='Validation frequency (epochs).')
	parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
	parser.add_argument('--logdir', type=str, default='log', help='Directory to log data to.')
	parser.add_argument('--gpus', type=int, default=1, help='number of gpus')

	args = parser.parse_args()
	args.logdir = os.path.join(args.logdir, args.id)

	# Config
	config = GlobalConfig()

	# Data
	train_set = CARLA_Data(root=config.root_dir_all, data_folders=config.train_data, img_aug = config.img_aug)
	#print(len(train_set))
	val_set = CARLA_Data(root=config.root_dir_all, data_folders=config.val_data)
	#print(len(val_set))

	dataloader_train = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=8)
	dataloader_val = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=2)

	feature_extractor = FCResNet(
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
		num_data=len(train_set),
		train_batch_size=args.batch_size,
		ridge_penalty=1.0,
		feature_scale=None,
		mean_field_factor=None
	)
	loss_fn = F.mse_loss

	if torch.cuda.is_available():
		model = model.cuda()

	print(model)

	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

	pbar = ProgressBar()
	train_loss = [] 

	def training_step(engine, batch):
		model.train()
		
		optimizer.zero_grad()

		front_img = batch['front_img']
		#front_img = cv2.cvtColor(front_img, cv2.COLOR_BGR2YUV)
		speed = batch['speed'].to(dtype=torch.float32).view(-1,1) / 12.
		target_point = batch['target_point'].to(dtype=torch.float32)
		command = batch['target_command']
		
		state = torch.cat([speed, target_point, command], 1)
		if torch.cuda.is_available():
			state = state.cuda()
			front_img = front_img.cuda()
			target_point = target_point.cuda()

		pred = model(front_img, state)

		acc_pred = pred['pred_accel'].unsqueeze(dim=1).cuda()
		brake_pred = pred['pred_brake'].unsqueeze(dim=1).cuda()
		#brake = torch.zeros_like(acc_pred)
		#for a in range(acc_pred.shape[0]):
			#if acc_pred[a] < 0:
				#brake[a] = abs(acc_pred[a])
		acc_true = batch['action'][:, 0].reshape(-1, 1).cuda()
		steer_pred = pred['pred_steer'].cuda()
		steer_true = batch['action'][:, 1].reshape(-1, 1).cuda()
		brake_true = batch['action'][:, 2].reshape(-1, 1).cuda()

		accel_loss = loss_fn(acc_pred, acc_true)
		theta_loss = loss_fn(steer_pred, steer_true)
		brake_loss = loss_fn(brake_pred, brake_true)
		loss = accel_loss + theta_loss + brake_loss

		train_loss.append([
			str(trainer.state.epoch), 
			str(accel_loss.item()), 
			str(theta_loss.item()), 
			str(brake_loss.item()), 
			str(loss.item())])

		loss.backward()
		optimizer.step()

		return loss.item()

	def validation_step(engine, batch):
		model.eval()

		front_img = batch['front_img']
		#front_img = cv2.cvtColor(front_img, cv2.COLOR_BGR2YUV)
		speed = batch['speed'].to(dtype=torch.float32).view(-1,1) / 12.
		target_point = batch['target_point'].to(dtype=torch.float32)
		command = batch['target_command']

		state = torch.cat([speed, target_point, command], 1)
		y = torch.cat([
			batch['action'][:, 1].reshape(-1, 1), 
			batch['action'][:, 0].reshape(-1, 1), 
			batch['action'][:, 2].reshape(-1, 1)], 1)
		if torch.cuda.is_available():
			state = state.cuda()
			front_img = front_img.cuda()
			target_point = target_point.cuda()
			y = y.cuda()

		y_pred = model(front_img, state)
		#brake = torch.zeros_like(y_pred['pred_accel'])
		#acc =y_pred['pred_accel'].squeeze()
		#for a in range(acc.shape[0]):
			#if acc[a] < 0:
				#brake[a] = abs(acc[a])
		#y_pred['brake'] = brake
		return y_pred, y

	trainer = Engine(training_step)
	evaluator = Engine(validation_step)

	metric = Average()
	metric.attach(trainer, "loss")
	pbar.attach(trainer)

	metric = Loss(lambda y_pred, y: 
			   F.mse_loss(
				   y_pred['pred_steer'].squeeze(), 
				   y[:, 0]) + 
				F.mse_loss(
					y_pred['pred_accel'], 
					y[:, 1]) +
				F.mse_loss(
					y_pred['pred_brake'], 
					y[:, 2]))
	metric.attach(evaluator, "loss")

	to_save = {'model': model}
	handler = Checkpoint(
		to_save, args.logdir,
		n_saved=None, filename_prefix='best',
		score_name="loss",
		global_step_transform=global_step_from_engine(trainer)
	)

	trainer.add_event_handler(Events.EPOCH_COMPLETED(every=int(args.epochs/10)), handler)
	trainer.add_event_handler(Events.COMPLETED, handler)

	loss_array = []

	@trainer.on(Events.EPOCH_COMPLETED(every=int(args.epochs/10)))
	def log_results(trainer):
		evaluator.run(dataloader_val)
		loss_array.append([
			str(trainer.state.epoch),
			str(trainer.state.metrics['loss']), 
			str(evaluator.state.metrics['loss'])])
		print(f"Results - Epoch: {trainer.state.epoch} - "
			f"Test Likelihood: {evaluator.state.metrics['loss']:.2f} - "
			f"Loss: {trainer.state.metrics['loss']:.2f}")
		
	@trainer.on(Events.EPOCH_COMPLETED)
	def show_results(trainer):
		print(f"Results - Epoch: {trainer.state.epoch} - "
			f"Loss: {trainer.state.metrics['loss']:.2f}")

	@trainer.on(Events.COMPLETED)
	def log_losses(trainer):
		with open(os.path.join(args.logdir, "train_loss_array-3.txt"), "w") as txt_file:
			for line in train_loss:
				txt_file.write(" ".join(line) + "\n") 
		with open(os.path.join(args.logdir, "val_loss_array-3.txt"), "w") as txt_file:
			for line in loss_array:
				txt_file.write(" ".join(line) + "\n") 

	trainer.run(dataloader_train, max_epochs=args.epochs)
