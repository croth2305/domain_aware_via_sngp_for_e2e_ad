import os
import json
import datetime
import pathlib
import time
import cv2
import carla
from collections import deque
import math
from collections import OrderedDict

import torch
import carla
import numpy as np
from PIL import Image
from torchvision import transforms as T

from leaderboard.autoagents import autonomous_agent

from DS.model import DAVE2_SNGP, FCResNet
from DS.config import GlobalConfig
from team_code.planner import RoutePlanner


SAVE_PATH = os.environ.get('SAVE_PATH', None)


def get_entry_point():
	return 'DAVE2_SNGP_Agent'


class DAVE2_SNGP_Agent(autonomous_agent.AutonomousAgent):
	def setup(self, path_to_conf_file):
		self.track = autonomous_agent.Track.SENSORS
		self.alpha = 0.3
		self.status = 0
		self.steer_step = 0
		self.last_moving_status = 0
		self.last_moving_step = -1
		self.last_steers = deque()

		self.config_path = path_to_conf_file
		self.step = -1
		self.wall_start = time.time()
		self.initialized = False

		self.config = GlobalConfig()
		self.fe = FCResNet(
			features=self.config.features,
			spectral_normalization=self.config.spectral_normalization,
			coeff=self.config.coeff,
			n_power_iterations=self.config.n_power_iterations,
			dropout_rate=self.config.dropout_rate
		)
		self.net = DAVE2_SNGP(
			feature_extractor=self.fe,
        	num_deep_features=self.config.features,
        	num_gp_features=self.config.num_gp_features,
        	normalize_gp_features=self.config.normalize_gp_features,
        	num_random_features=self.config.num_random_features,
        	num_outputs=self.config.num_outputs,
        	num_data=1, #TODO
        	train_batch_size=self.config.train_batch_size,
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
			self.net.load_state_dict(new_state_dict, strict = False)
		except:
			try:
				checkpoint = torch.load(path_to_conf_file)
				self.net.load_state_dict(checkpoint)
			except Exception as e:
				print(e)
	
		self.net.cuda()
		self.net.eval()

		self.takeover = False
		self.stop_time = 0
		self.takeover_time = 0

		self.save_path = None
		self._im_transform = T.Compose([T.ToTensor(), T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])

		self.last_steers = deque()
		if SAVE_PATH is not None:
			now = datetime.datetime.now()
			string = pathlib.Path(os.environ['ROUTES']).stem + '_'
			string += '_'.join(map(lambda x: '%02d' % x, (now.month, now.day, now.hour, now.minute, now.second)))

			print (string)

			self.save_path = pathlib.Path(os.environ['SAVE_PATH']) / string
			self.save_path.mkdir(parents=True, exist_ok=False)

			(self.save_path / 'rgb').mkdir()
			(self.save_path / 'meta').mkdir()
			(self.save_path / 'bev').mkdir()
			(self.save_path / 'fms').mkdir()

	def _init(self):
		self._route_planner = RoutePlanner(4.0, 50.0)
		self._route_planner.set_route(self._global_plan, True)

		self.initialized = True

	def _get_position(self, tick_data):
		gps = tick_data['gps']
		gps = (gps - self._route_planner.mean) * self._route_planner.scale

		return gps

	def sensors(self):
				return [
				{
					'type': 'sensor.camera.rgb',
					'x': -1.5, 'y': 0.0, 'z':2.0,
					'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
					'width': 900, 'height': 256, 'fov': 100,
					'id': 'rgb'
					},
				{
					'type': 'sensor.camera.rgb',
					'x': 0.0, 'y': 0.0, 'z': 50.0,
					'roll': 0.0, 'pitch': -90.0, 'yaw': 0.0,
					'width': 512, 'height': 512, 'fov': 5 * 10.0,
					'id': 'bev'
					},	
				{
					'type': 'sensor.other.imu',
					'x': 0.0, 'y': 0.0, 'z': 0.0,
					'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
					'sensor_tick': 0.05,
					'id': 'imu'
					},
				{
					'type': 'sensor.other.gnss',
					'x': 0.0, 'y': 0.0, 'z': 0.0,
					'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
					'sensor_tick': 0.01,
					'id': 'gps'
					},
				{
					'type': 'sensor.speedometer',
					'reading_frequency': 20,
					'id': 'speed'
					}
				]

	def tick(self, input_data):
		self.step += 1

		rgb = cv2.cvtColor(input_data['rgb'][1][:, :, :3], cv2.COLOR_BGR2RGB)
		bev = cv2.cvtColor(input_data['bev'][1][:, :, :3], cv2.COLOR_BGR2RGB)
		gps = input_data['gps'][1][:2]
		speed = input_data['speed'][1]['speed']
		compass = input_data['imu'][1][-1]

		if (math.isnan(compass) == True): #It can happen that the compass sends nan for a few frames
			compass = 0.0

		result = {
				'rgb': rgb,
				'gps': gps,
				'speed': speed,
				'compass': compass,
				'bev': bev
				}
		
		pos = self._get_position(result)
		result['gps'] = pos
		next_wp, next_cmd = self._route_planner.run_step(pos)
		result['next_command'] = next_cmd.value


		theta = compass + np.pi/2
		R = np.array([
			[np.cos(theta), -np.sin(theta)],
			[np.sin(theta), np.cos(theta)]
			])

		local_command_point = np.array([next_wp[0]-pos[0], next_wp[1]-pos[1]])
		local_command_point = R.T.dot(local_command_point)
		result['target_point'] = tuple(local_command_point)

		return result
	@torch.no_grad()
	def run_step(self, input_data, timestamp):
		if not self.initialized:
			self._init()
		tick_data = self.tick(input_data)
		if self.step < self.config.seq_len:
			rgb = self._im_transform(tick_data['rgb']).unsqueeze(0)

			control = carla.VehicleControl()
			control.steer = 0.0
			control.throttle = 0.0
			control.brake = 0.0
			
			return control

		gt_velocity = torch.FloatTensor([tick_data['speed']]).to('cuda', dtype=torch.float32)
		command = tick_data['next_command']
		if command < 0:
			command = 4
		command -= 1
		assert command in [0, 1, 2, 3, 4, 5]
		cmd_one_hot = [0] * 6
		cmd_one_hot[command] = 1 
		cmd_one_hot = torch.tensor(cmd_one_hot).view(1, 6).to('cuda', dtype=torch.float32)
		speed = torch.FloatTensor([float(tick_data['speed'])]).view(1,1).to('cuda', dtype=torch.float32)
		speed = speed / 12
		rgb = self._im_transform(tick_data['rgb']).unsqueeze(0).to('cuda', dtype=torch.float32)

		tick_data['target_point'] = [torch.FloatTensor([tick_data['target_point'][0]]),
										torch.FloatTensor([tick_data['target_point'][1]])]
		target_point = torch.stack(tick_data['target_point'], dim=1).to('cuda', dtype=torch.float32)
		state = torch.cat([speed, target_point, cmd_one_hot], 1)

		pred = self.net(rgb, state)
		self.metadata = {} 

		control = carla.VehicleControl()
		s = float(pred['pred_steer'].cuda().item())
		a = float(pred['pred_accel'].cuda().item())
		b = float(pred['pred_brake'].cuda().item())
		uncertainty_s = pred['variance_steer'].cuda().item()
		uncertainty_a = pred['variance_accel'].cuda().item()
		self.feature_maps = pred['feature_map']

		if a > b:
			b = 0.0
		else:
			a = 0.0

		control.steer = s
		control.throttle = a
		control.brake = b

		if control.brake > 0.5:
			control.throttle = float(0)

		#print(f"s: {control.steer} a: {control.throttle} b: {control.brake}")
		
		self.metadata['steer'] = float(s)
		self.metadata['uncertainty_steer'] = float(uncertainty_s)
		self.metadata['throttle'] = float(a)
		self.metadata['uncertainty_throttle'] = float(uncertainty_a)
		self.metadata['brake'] = float(b)
		self.metadata['position x'] = tick_data['gps'][0] 
		self.metadata['position y'] = tick_data['gps'][1] 
		self.metadata['compass'] = str(tick_data['compass'])
		state = state.squeeze()
		self.metadata['state_0'] = float(state[0].item())
		self.metadata['state_1'] = float(state[1].item())
		self.metadata['state_2'] = float(state[2].item())
		self.metadata['state_3'] = int(state[3].item())
		self.metadata['state_4'] = int(state[4].item())
		self.metadata['state_5'] = int(state[5].item())
		self.metadata['state_6'] = int(state[6].item())
		self.metadata['state_7'] = int(state[7].item())
		self.metadata['state_8'] = int(state[8].item())

		if SAVE_PATH is not None and self.step % 10 == 0:
			self.save(tick_data)
		return control

	def save(self, tick_data):
		frame = self.step // 10

		Image.fromarray(tick_data['rgb']).save(self.save_path / 'rgb' / ('%04d.png' % frame))

		Image.fromarray(tick_data['bev']).save(self.save_path / 'bev' / ('%04d.png' % frame))

		outfile = open(self.save_path / 'meta' / ('%04d.json' % frame), 'w')
		json.dump(self.metadata, outfile, indent=4)
		outfile.close()

		torch.save(self.feature_maps, os.path.join(self.save_path / 'fms' / ('%04d.pt' % frame)))

	def destroy(self):
		del self.net
		torch.cuda.empty_cache()