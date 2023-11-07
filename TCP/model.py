from collections import deque
import numpy as np
import torch 
from torch import nn
from TCP.resnet import *

from TCP.monodepth2 import MonodepthModel

import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import splprep, splev
import sympy as sym
from sympy.tensor.array import derive_by_array
sym.init_printing()

from TCP.utils.abstract_controller import Controller



class PIDController(object):
	def __init__(self, K_P=1.0, K_I=0.0, K_D=0.0, n=20):
		self._K_P = K_P
		self._K_I = K_I
		self._K_D = K_D

		self._window = deque([0 for _ in range(n)], maxlen=n)
		self._max = 0.0
		self._min = 0.0

	def step(self, error):
		self._window.append(error)
		self._max = max(self._max, abs(error))
		self._min = -abs(self._max)

		if len(self._window) >= 2:
			integral = np.mean(self._window)
			derivative = (self._window[-1] - self._window[-2])
		else:
			integral = 0.0
			derivative = 0.0

		return self._K_P * error + self._K_I * integral + self._K_D * derivative


class _EqualityConstraints(object):
	"""Class for storing equality constraints in the MPC."""
	def __init__(self, N, state_vars):
		self.dict = {}
		for symbol in state_vars:
			self.dict[symbol] = N*[None]

	def __getitem__(self, key):
		return self.dict[key]

	def __setitem__(self, key, value):
		self.dict[key] = value

class MPCController(Controller):
	def __init__(self, config, target_speed, steps_ahead=10, dt=0.1):
		self.target_speed = target_speed
		self.state_vars = ('x', 'y', 'v', 'ψ', 'cte', 'eψ')
		self.config = config

		self.steps_ahead = steps_ahead
		self.dt = dt

		# Cost function coefficients
		self.cte_coeff = 100 # 100
		self.epsi_coeff = 100 # 100
		self.speed_coeff = 0.4  # 0.2
		self.acc_coeff = 1  # 1
		self.steer_coeff = 0.1  # 0.1
		self.consec_acc_coeff = 50
		self.consec_steer_coeff = 50

		# Front wheel L
		self.Lf = 2.5  # TODO: check if true

		# How the polynomial fitting the desired curve is fitted
		self.steps_poly = 30
		self.poly_degree = 3

		# Bounds for the optimizer
		self.bounds = (
			6*self.steps_ahead * [(None, None)]
			+ self.steps_ahead * [self.config.THROTTLE_BOUNDS]
			+ self.steps_ahead * [self.config.STEER_BOUNDS]
		)

		# State 0 placeholder
		num_vars = (len(self.state_vars) + 2)  # State variables and two actuators
		self.state0 = np.zeros(self.steps_ahead*num_vars)

		# Lambdify and minimize stuff
		self.evaluator = 'numpy'
		self.tolerance = 1
		self.cost_func, self.cost_grad_func, self.constr_funcs = self.get_func_constraints_and_bounds()

		# To keep the previous state
		self.steer = None
		self.throttle = None

	def get_func_constraints_and_bounds(self):
		"""The most important method of this class, defining the MPC's cost
		function and constraints.
		"""
		# Polynomial coefficients will also be symbolic variables
		poly = self.create_array_of_symbols('poly', self.poly_degree+1)

		# Initialize the initial state
		x_init = sym.symbols('x_init')
		y_init = sym.symbols('y_init')
		ψ_init = sym.symbols('ψ_init')
		v_init = sym.symbols('v_init')
		cte_init = sym.symbols('cte_init')
		eψ_init = sym.symbols('eψ_init')

		init = (x_init, y_init, ψ_init, v_init, cte_init, eψ_init)

		# State variables
		x = self.create_array_of_symbols('x', self.steps_ahead)
		y = self.create_array_of_symbols('y', self.steps_ahead)
		ψ = self.create_array_of_symbols('ψ', self.steps_ahead)
		v = self.create_array_of_symbols('v', self.steps_ahead)
		cte = self.create_array_of_symbols('cte', self.steps_ahead)
		eψ = self.create_array_of_symbols('eψ', self.steps_ahead)

		# Actuators
		a = self.create_array_of_symbols('a', self.steps_ahead)
		δ = self.create_array_of_symbols('δ', self.steps_ahead)

		vars_ = (
			# Symbolic arrays (but NOT actuators)
			*x, *y, *ψ, *v, *cte, *eψ,

			# Symbolic arrays (actuators)
			*a, *δ,
		)

		cost = 0
		for t in range(self.steps_ahead):
			cost += (
				# Reference state penalties
				self.cte_coeff * cte[t]**2
				+ self.epsi_coeff * eψ[t]**2 +
				+ self.speed_coeff * (v[t] - self.target_speed)**2

				# # Actuator penalties
				+ self.acc_coeff * a[t]**2
				+ self.steer_coeff * δ[t]**2
			)

		# Penalty for differences in consecutive actuators
		for t in range(self.steps_ahead-1):
			cost += (
				self.consec_acc_coeff * (a[t+1] - a[t])**2
				+ self.consec_steer_coeff * (δ[t+1] - δ[t])**2
			)

		# Initialize constraints
		eq_constr = _EqualityConstraints(self.steps_ahead, self.state_vars)
		eq_constr['x'][0] = x[0] - x_init
		eq_constr['y'][0] = y[0] - y_init
		eq_constr['ψ'][0] = ψ[0] - ψ_init
		eq_constr['v'][0] = v[0] - v_init
		eq_constr['cte'][0] = cte[0] - cte_init
		eq_constr['eψ'][0] = eψ[0] - eψ_init

		for t in range(1, self.steps_ahead):
			curve = sum(poly[-(i+1)] * x[t-1]**i for i in range(len(poly)))
			# The desired ψ is equal to the derivative of the polynomial curve at
			#  point x[t-1]
			ψdes = sum(poly[-(i+1)] * i*x[t-1]**(i-1) for i in range(1, len(poly)))

			eq_constr['x'][t] = x[t] - (x[t-1] + v[t-1] * sym.cos(ψ[t-1]) * self.dt)
			eq_constr['y'][t] = y[t] - (y[t-1] + v[t-1] * sym.sin(ψ[t-1]) * self.dt)
			eq_constr['ψ'][t] = ψ[t] - (ψ[t-1] - v[t-1] * δ[t-1] / self.Lf * self.dt)
			eq_constr['v'][t] = v[t] - (v[t-1] + a[t-1] * self.dt)
			eq_constr['cte'][t] = cte[t] - (curve - y[t-1] + v[t-1] * sym.sin(eψ[t-1]) * self.dt)
			eq_constr['eψ'][t] = eψ[t] - (ψ[t-1] - ψdes - v[t-1] * δ[t-1] / self.Lf * self.dt)

		# Generate actual functions from
		cost_func = self.generate_fun(cost, vars_, init, poly)
		cost_grad_func = self.generate_grad(cost, vars_, init, poly)

		constr_funcs = []
		for symbol in self.state_vars:
			for t in range(self.steps_ahead):
				func = self.generate_fun(eq_constr[symbol][t], vars_, init, poly)
				grad_func = self.generate_grad(eq_constr[symbol][t], vars_, init, poly)
				constr_funcs.append(
					{'type': 'eq', 'fun': func, 'jac': grad_func, 'args': None},
				)

		return cost_func, cost_grad_func, constr_funcs


	def control(self, pts_2D, measurements):
		pts_2D = pts_2D[0].data.cpu().numpy()
		which_closest, _, location = self._calc_closest_dists_and_location(
			measurements,
			pts_2D
		)

		# Stabilizes polynomial fitting
		which_closest_shifted = which_closest - 5
		# NOTE: `which_closest_shifted` might become < 0, but the modulo operation below fixes that

		indeces = which_closest_shifted + self.steps_poly*np.arange(self.poly_degree+1)
		indeces = indeces % pts_2D.shape[0]
		pts = pts_2D[indeces]

		waypoints = pts_2D
		num_pairs = len(waypoints) - 1
		target = measurements['target'].squeeze().data.cpu().numpy()

		# flip y (forward is negative in our waypoints)
		waypoints[:,1] *= -1
		target[1] *= -1

		# iterate over vectors between predicted waypoints
		num_pairs = len(waypoints) - 1
		best_norm = 1e5
		desired_speed = 0

		aim = waypoints[0]
		for i in range(num_pairs):
			# magnitude of vectors, used for speed
			desired_speed += np.linalg.norm(
					waypoints[i+1] - waypoints[i]) * 2.0 / num_pairs

			# norm of vector midpoints, used for steering
			norm = np.linalg.norm((waypoints[i+1] + waypoints[i]) / 2.0)
			if abs(self.config.aim_dist-best_norm) > abs(self.config.aim_dist-norm):
				aim = waypoints[i]
				best_norm = norm

		aim_last = waypoints[-1] - waypoints[-2]

		angle = np.degrees(np.pi / 2 - np.arctan2(aim[1], aim[0])) / 90
		angle_last = np.degrees(np.pi / 2 - np.arctan2(aim_last[1], aim_last[0])) / 90
		angle_target = np.degrees(np.pi / 2 - np.arctan2(target[1], target[0])) / 90

		# choice of point to aim for steering, removing outlier predictions
		# use target point if it has a smaller angle or if error is large
		# predicted point otherwise
		# (reduces noise in eg. straight roads, helps with sudden turn commands)
		use_target_to_aim = np.abs(angle_target) < np.abs(angle)
		use_target_to_aim = use_target_to_aim or (np.abs(angle_target-angle_last) > self.config.angle_thresh and target[1] < self.config.dist_thresh)
		if use_target_to_aim:
			angle_final = angle_target
		else:
			angle_final = angle


		# orient = measurements.player_measurements.transform.orientation
		# v = measurements.player_measurements.forward_speed * 3.6 # km / h
		v = measurements['speed'].data.cpu().numpy() * 3.6 # km / h
		ψ = angle_final
		# print(measurements, pts_2D)

		cos_ψ = np.cos(ψ)
		sin_ψ = np.sin(ψ)

		x, y = location[0], location[1]
		pts_car = MPCController.transform_into_cars_coordinate_system(pts, x, y, cos_ψ, sin_ψ)

		poly = np.polyfit(pts_car[:, 0], pts_car[:, 1], self.poly_degree)

		cte = poly[-1]
		eψ = -np.arctan(poly[-2])

		init = (0, 0, 0, v, cte, eψ, *poly)
		self.state0 = self.get_state0(v, cte, eψ, self.steer, self.throttle, poly)
		result = self.minimize_cost(self.bounds, self.state0, init)

		# Left here for debugging
		# self.steer = -0.6 * cte - 5.5 * (cte - prev_cte)
		# prev_cte = cte
		# self.throttle = clip_throttle(self.throttle, v, target_speed)

		if 'success' in result.message:
			self.steer = result.x[-self.steps_ahead]
			self.throttle = result.x[-2*self.steps_ahead]
		else:
			print('Unsuccessful optimization')
		
		brake = self.target_speed < self.config.brake_speed or (v / self.target_speed) > self.config.brake_ratio

		one_log_dict = {
			'speed': float(v.astype(np.float64)),
			'steer': float(self.steer),
			'throttle': float(self.throttle),
			'break' : float(brake),
			'psi': float(ψ),
			'cte': float(cte),
			'epsi': float(eψ),
			# 'which_closest': which_closest,
			'x': float(x),
			'y': float(y),
		}
		for i, coeff in enumerate(poly):
			one_log_dict['poly{}'.format(i)] = coeff

		for i in range(pts_car.shape[0]):
			for j in range(pts_car.shape[1]):
				one_log_dict['pts_car_{}_{}'.format(i, j)] = pts_car[i][j]

		return self.steer, self.throttle, brake, one_log_dict

	def get_state0(self, v, cte, epsi, a, delta, poly):
		a = a or 0
		delta = delta or 0
		# "Go as the road goes"
		# x = np.linspace(0, self.steps_ahead*self.dt*v, self.steps_ahead)
		# y = np.polyval(poly, x)
		x = np.linspace(0, 1, self.steps_ahead)
		y = np.polyval(poly, x)
		psi = 0

		self.state0[:self.steps_ahead] = x
		self.state0[self.steps_ahead:2*self.steps_ahead] = y
		self.state0[2*self.steps_ahead:3*self.steps_ahead] = psi
		self.state0[3*self.steps_ahead:4*self.steps_ahead] = v
		self.state0[4*self.steps_ahead:5*self.steps_ahead] = cte
		self.state0[5*self.steps_ahead:6*self.steps_ahead] = epsi
		self.state0[6*self.steps_ahead:7*self.steps_ahead] = a
		self.state0[7*self.steps_ahead:8*self.steps_ahead] = delta
		return self.state0

	def generate_fun(self, symb_fun, vars_, init, poly):
		'''This function generates a function of the form `fun(x, *args)` because
		that's what the scipy `minimize` API expects (if we don't want to minimize
		over certain variables, we pass them as `args`)
		'''
		args = init + poly
		return sym.lambdify((vars_, *args), symb_fun, self.evaluator)
		# Equivalent to (but faster than):
		# func = sym.lambdify(vars_+init+poly, symb_fun, evaluator)
		# return lambda x, *args: func(*np.r_[x, args])

	def generate_grad(self, symb_fun, vars_, init, poly):
		args = init + poly
		return sym.lambdify(
			(vars_, *args),
			derive_by_array(symb_fun, vars_+args)[:len(vars_)],
			self.evaluator
		)
		# Equivalent to (but faster than):
		# cost_grad_funcs = [
		#     generate_fun(symb_fun.diff(var), vars_, init, poly)
		#     for var in vars_
		# ]
		# return lambda x, *args: [
		#     grad_func(np.r_[x, args]) for grad_func in cost_grad_funcs
		# ]

	def minimize_cost(self, bounds, x0, init):
		# TODO: this is a bit retarded, but hey -- that's scipy API's fault ;)
		for constr_func in self.constr_funcs:
			constr_func['args'] = init

		return minimize(
			fun=self.cost_func,
			x0=x0,
			args=init,
			jac=self.cost_grad_func,
			bounds=bounds,
			constraints=self.constr_funcs,
			method='SLSQP',
			tol=self.tolerance,
		)

	@staticmethod
	def create_array_of_symbols(str_symbol, N):
		return sym.symbols('{symbol}0:{N}'.format(symbol=str_symbol, N=N))

	@staticmethod
	def transform_into_cars_coordinate_system(pts, x, y, cos_ψ, sin_ψ):
		diff = (pts - [x, y])
		pts_car = np.zeros_like(diff)
		pts_car[:, 0] = cos_ψ * diff[:, 0] + sin_ψ * diff[:, 1]
		pts_car[:, 1] = sin_ψ * diff[:, 0] - cos_ψ * diff[:, 1]
		return pts_car
	
	
class TCP(nn.Module):

	def __init__(self, config):
		super().__init__()
		self.config = config

		self.turn_controller = PIDController(K_P=config.turn_KP, K_I=config.turn_KI, K_D=config.turn_KD, n=config.turn_n)
		self.speed_controller = PIDController(K_P=config.speed_KP, K_I=config.speed_KI, K_D=config.speed_KD, n=config.speed_n)

		self.perception = resnet34(pretrained=True)
		self.depthmap = MonodepthModel(use_gpu=True)

		self.measurements = nn.Sequential(
							nn.Linear(1+2+6, 128),
							nn.ReLU(inplace=True),
							nn.Linear(128, 128),
							nn.ReLU(inplace=True),
						)

		self.join_traj = nn.Sequential(
							nn.Linear(128+1000, 512),
							nn.ReLU(inplace=True),
							nn.Linear(512, 512),
							nn.ReLU(inplace=True),
							nn.Linear(512, 256),
							nn.ReLU(inplace=True),
						)

		self.join_ctrl = nn.Sequential(
							nn.Linear(128+512, 512),
							nn.ReLU(inplace=True),
							nn.Linear(512, 512),
							nn.ReLU(inplace=True),
							nn.Linear(512, 256),
							nn.ReLU(inplace=True),
						)

		self.speed_branch = nn.Sequential(
							nn.Linear(1000, 256),
							nn.ReLU(inplace=True),
							nn.Linear(256, 256),
							nn.Dropout2d(p=0.5),
							nn.ReLU(inplace=True),
							nn.Linear(256, 1),
						)

		self.value_branch_traj = nn.Sequential(
					nn.Linear(256, 256),
					nn.ReLU(inplace=True),
					nn.Linear(256, 256),
					nn.Dropout2d(p=0.5),
					nn.ReLU(inplace=True),
					nn.Linear(256, 1),
				)
		self.value_branch_ctrl = nn.Sequential(
					nn.Linear(256, 256),
					nn.ReLU(inplace=True),
					nn.Linear(256, 256),
					nn.Dropout2d(p=0.5),
					nn.ReLU(inplace=True),
					nn.Linear(256, 1),
				)
		# shared branches_neurons
		dim_out = 2

		self.policy_head = nn.Sequential(
				nn.Linear(256, 256),
				nn.ReLU(inplace=True),
				nn.Linear(256, 256),
				nn.Dropout2d(p=0.5),
				nn.ReLU(inplace=True),
			)
		self.decoder_ctrl = nn.GRUCell(input_size=256+4, hidden_size=256)
		self.output_ctrl = nn.Sequential(
				nn.Linear(256, 256),
				nn.ReLU(inplace=True),
				nn.Linear(256, 256),
				nn.ReLU(inplace=True),
			)
		self.dist_mu = nn.Sequential(nn.Linear(256, dim_out), nn.Softplus())
		self.dist_sigma = nn.Sequential(nn.Linear(256, dim_out), nn.Softplus())


		self.decoder_traj = nn.GRUCell(input_size=4, hidden_size=256)
		self.output_traj = nn.Linear(256, 2)

		self.init_att = nn.Sequential(
				nn.Linear(128, 256),
				nn.ReLU(inplace=True),
				nn.Linear(256, 29*8),
				nn.Softmax(1)
			)

		self.wp_att = nn.Sequential(
				nn.Linear(256+256, 256),
				nn.ReLU(inplace=True),
				nn.Linear(256, 29*8),
				nn.Softmax(1)
			)

		self.merge = nn.Sequential(
				nn.Linear(512+256, 512),
				nn.ReLU(inplace=True),
				nn.Linear(512, 256),
			)
		

	def forward(self, img, state, target_point):
		feature_emb, cnn_feature = self.perception(img)
		# Feature embeddings : torch.Size([32, 1000])
		# CNN features: torch.Size([32, 512, 8, 29])
		# features, depth_features = self.depthmap.predict_depth_batch(img_o)
		# depth features: torch.Size([32, 192, 640])
		outputs = {}
		outputs['pred_speed'] = self.speed_branch(feature_emb)
		measurement_feature = self.measurements(state)
		
		j_traj = self.join_traj(torch.cat([feature_emb, measurement_feature], 1))
		outputs['pred_value_traj'] = self.value_branch_traj(j_traj)
		outputs['pred_features_traj'] = j_traj
		z = j_traj
		output_wp = list()
		traj_hidden_state = list()

		# initial input variable to GRU
		x = torch.zeros(size=(z.shape[0], 2), dtype=z.dtype).type_as(z)

		# autoregressive generation of output waypoints
		for _ in range(self.config.pred_len):
			x_in = torch.cat([x, target_point], dim=1)
			z = self.decoder_traj(x_in, z)
			traj_hidden_state.append(z)
			dx = self.output_traj(z)
			x = dx + x
			output_wp.append(x)

		pred_wp = torch.stack(output_wp, dim=1)
		outputs['pred_wp'] = pred_wp

		traj_hidden_state = torch.stack(traj_hidden_state, dim=1)
		init_att = self.init_att(measurement_feature).view(-1, 1, 8, 29)
		feature_emb = torch.sum(cnn_feature*init_att, dim=(2, 3))
		j_ctrl = self.join_ctrl(torch.cat([feature_emb, measurement_feature], 1))
		outputs['pred_value_ctrl'] = self.value_branch_ctrl(j_ctrl)
		outputs['pred_features_ctrl'] = j_ctrl
		policy = self.policy_head(j_ctrl)
		outputs['mu_branches'] = self.dist_mu(policy)
		outputs['sigma_branches'] = self.dist_sigma(policy)

		x = j_ctrl
		mu = outputs['mu_branches']
		sigma = outputs['sigma_branches']
		future_feature, future_mu, future_sigma = [], [], []

		# initial hidden variable to GRU
		h = torch.zeros(size=(x.shape[0], 256), dtype=x.dtype).type_as(x)

		for _ in range(self.config.pred_len):
			x_in = torch.cat([x, mu, sigma], dim=1)
			h = self.decoder_ctrl(x_in, h)
			wp_att = self.wp_att(torch.cat([h, traj_hidden_state[:, _]], 1)).view(-1, 1, 8, 29)
			new_feature_emb = torch.sum(cnn_feature*wp_att, dim=(2, 3))
			merged_feature = self.merge(torch.cat([h, new_feature_emb], 1))
			dx = self.output_ctrl(merged_feature)
			x = dx + x

			policy = self.policy_head(x)
			mu = self.dist_mu(policy)
			sigma = self.dist_sigma(policy)
			future_feature.append(x)
			future_mu.append(mu)
			future_sigma.append(sigma)


		outputs['future_feature'] = future_feature
		outputs['future_mu'] = future_mu
		outputs['future_sigma'] = future_sigma
		return outputs

	def process_mpc(self, waypoints, measurements, speed):
		mpc = MPCController(self.config, speed)
		return mpc.control(waypoints, measurements)

	def process_action(self, pred, command, speed, target_point):
		action = self._get_action_beta(pred['mu_branches'].view(1,2), pred['sigma_branches'].view(1,2))
		acc, steer = action.cpu().numpy()[0].astype(np.float64)
		if acc >= 0.0:
			throttle = acc
			brake = 0.0
		else:
			throttle = 0.0
			brake = np.abs(acc)

		throttle = np.clip(throttle, 0, 1)
		steer = np.clip(steer, -1, 1)
		brake = np.clip(brake, 0, 1)

		metadata = {
			'speed': float(speed.cpu().numpy().astype(np.float64)),
			'steer': float(steer),
			'throttle': float(throttle),
			'brake': float(brake),
			'command': command,
			'target_point': tuple(target_point[0].data.cpu().numpy().astype(np.float64)),
		}
		return steer, throttle, brake, metadata

	def _get_action_beta(self, alpha, beta):
		x = torch.zeros_like(alpha)
		x[:, 1] += 0.5
		mask1 = (alpha > 1) & (beta > 1)
		x[mask1] = (alpha[mask1]-1)/(alpha[mask1]+beta[mask1]-2)

		mask2 = (alpha <= 1) & (beta > 1)
		x[mask2] = 0.0

		mask3 = (alpha > 1) & (beta <= 1)
		x[mask3] = 1.0

		# mean
		mask4 = (alpha <= 1) & (beta <= 1)
		x[mask4] = alpha[mask4]/torch.clamp((alpha[mask4]+beta[mask4]), min=1e-5)

		x = x * 2 - 1

		return x

	def control_pid(self, waypoints, velocity, target):
		''' Predicts vehicle control with a PID controller.
		Args:
			waypoints (tensor): output of self.plan()
			velocity (tensor): speedometer input
		'''
		assert(waypoints.size(0)==1)
		waypoints = waypoints[0].data.cpu().numpy()
		target = target.squeeze().data.cpu().numpy()

		# flip y (forward is negative in our waypoints)
		waypoints[:,1] *= -1
		target[1] *= -1

		# iterate over vectors between predicted waypoints
		num_pairs = len(waypoints) - 1
		best_norm = 1e5
		desired_speed = 0
		aim = waypoints[0]
		for i in range(num_pairs):
			# magnitude of vectors, used for speed
			desired_speed += np.linalg.norm(
					waypoints[i+1] - waypoints[i]) * 2.0 / num_pairs

			# norm of vector midpoints, used for steering
			norm = np.linalg.norm((waypoints[i+1] + waypoints[i]) / 2.0)
			if abs(self.config.aim_dist-best_norm) > abs(self.config.aim_dist-norm):
				aim = waypoints[i]
				best_norm = norm

		aim_last = waypoints[-1] - waypoints[-2]

		angle = np.degrees(np.pi / 2 - np.arctan2(aim[1], aim[0])) / 90
		angle_last = np.degrees(np.pi / 2 - np.arctan2(aim_last[1], aim_last[0])) / 90
		angle_target = np.degrees(np.pi / 2 - np.arctan2(target[1], target[0])) / 90

		# choice of point to aim for steering, removing outlier predictions
		# use target point if it has a smaller angle or if error is large
		# predicted point otherwise
		# (reduces noise in eg. straight roads, helps with sudden turn commands)
		use_target_to_aim = np.abs(angle_target) < np.abs(angle)
		use_target_to_aim = use_target_to_aim or (np.abs(angle_target-angle_last) > self.config.angle_thresh and target[1] < self.config.dist_thresh)
		if use_target_to_aim:
			angle_final = angle_target
		else:
			angle_final = angle

		steer = self.turn_controller.step(angle_final)
		steer = np.clip(steer, -1.0, 1.0)

		speed = velocity[0].data.cpu().numpy()
		brake = desired_speed < self.config.brake_speed or (speed / desired_speed) > self.config.brake_ratio

		delta = np.clip(desired_speed - speed, 0.0, self.config.clip_delta)
		throttle = self.speed_controller.step(delta)
		throttle = np.clip(throttle, 0.0, self.config.max_throttle)
		throttle = throttle if not brake else 0.0

		metadata = {
			'speed': float(speed.astype(np.float64)),
			'steer': float(steer),
			'throttle': float(throttle),
			'brake': float(brake),
			'wp_4': tuple(waypoints[3].astype(np.float64)),
			'wp_3': tuple(waypoints[2].astype(np.float64)),
			'wp_2': tuple(waypoints[1].astype(np.float64)),
			'wp_1': tuple(waypoints[0].astype(np.float64)),
			'aim': tuple(aim.astype(np.float64)),
			'target': tuple(target.astype(np.float64)),
			'desired_speed': float(desired_speed.astype(np.float64)),
			'angle': float(angle.astype(np.float64)),
			'angle_last': float(angle_last.astype(np.float64)),
			'angle_target': float(angle_target.astype(np.float64)),
			'angle_final': float(angle_final.astype(np.float64)),
			'delta': float(delta.astype(np.float64)),
		}

		return steer, throttle, brake, metadata


	def get_action(self, mu, sigma):
		action = self._get_action_beta(mu.view(1,2), sigma.view(1,2))
		acc, steer = action[:, 0], action[:, 1]
		if acc >= 0.0:
			throttle = acc
			brake = torch.zeros_like(acc)
		else:
			throttle = torch.zeros_like(acc)
			brake = torch.abs(acc)

		throttle = torch.clamp(throttle, 0, 1)
		steer = torch.clamp(steer, -1, 1)
		brake = torch.clamp(brake, 0, 1)

		return throttle, steer, brake