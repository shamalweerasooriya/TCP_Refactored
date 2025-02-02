import argparse
import os
from collections import OrderedDict

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.distributions import Beta
import torchvision
import random
import wandb


import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger

from TCP.model import TCP
from TCP.data import CARLA_Data
from TCP.config import GlobalConfig


class TCP_planner(pl.LightningModule):
	def __init__(self, config, lr):
		super().__init__()
		self.lr = lr
		self.config = config
		self.model = TCP(config)
		self._load_weight()
		# self.save_hyperparameters()

	def _load_weight(self):
		# They are loading the state dict from roach .pth file
		rl_state_dict = torch.load(self.config.rl_ckpt, map_location='cpu')['policy_state_dict'] 
		# loading value_head weights and biases from roach state dict to the trajectory branch
		self._load_state_dict(self.model.value_branch_traj, rl_state_dict, 'value_head')
		# same as above for control look ckpt notebook for more info
		self._load_state_dict(self.model.value_branch_ctrl, rl_state_dict, 'value_head') 
		self._load_state_dict(self.model.dist_mu, rl_state_dict, 'dist_mu')
		self._load_state_dict(self.model.dist_sigma, rl_state_dict, 'dist_sigma')

	def _load_state_dict(self, il_net, rl_state_dict, key_word):
		rl_keys = [k for k in rl_state_dict.keys() if key_word in k]
		il_keys = il_net.state_dict().keys()
		assert len(rl_keys) == len(il_net.state_dict().keys()), f'mismatch number of layers loading {key_word}'
		new_state_dict = OrderedDict()
		for k_il, k_rl in zip(il_keys, rl_keys):
			new_state_dict[k_il] = rl_state_dict[k_rl]
		il_net.load_state_dict(new_state_dict)
	
	def forward(self, batch):
		pass

	def training_step(self, batch, batch_idx):
		front_img = batch['front_img']
		front_img_o = batch['front_img_o']
		speed = batch['speed'].to(dtype=torch.float32).view(-1,1) / 12.
		target_point = batch['target_point'].to(dtype=torch.float32)
		command = batch['target_command']
		
		state = torch.cat([speed, target_point, command], 1)
		value = batch['value'].view(-1,1)
		feature = batch['feature']

		gt_waypoints = batch['waypoints']

		pred = self.model(front_img, state, target_point)

		if(batch_idx==0):
			self.ref_front_img = front_img_o[0]
			self.ref_front_img_all = front_img_o
			self.ref_state = state[0] 
			self.ref_target_point = target_point[0] 
			print("ref_img_shape: ", self.ref_front_img.shape)

		dist_sup = Beta(batch['action_mu'], batch['action_sigma'])
		dist_pred = Beta(pred['mu_branches'], pred['sigma_branches'])
		kl_div = torch.distributions.kl_divergence(dist_sup, dist_pred)
		action_loss = torch.mean(kl_div[:, 0]) *0.5 + torch.mean(kl_div[:, 1]) *0.5
		speed_loss = F.l1_loss(pred['pred_speed'], speed) * self.config.speed_weight
		value_loss = (F.mse_loss(pred['pred_value_traj'], value) + F.mse_loss(pred['pred_value_ctrl'], value)) * self.config.value_weight
		feature_loss = (F.mse_loss(pred['pred_features_traj'], feature) +F.mse_loss(pred['pred_features_ctrl'], feature))* self.config.features_weight

		future_feature_loss = 0
		future_action_loss = 0
		for i in range(self.config.pred_len):
			dist_sup = Beta(batch['future_action_mu'][i], batch['future_action_sigma'][i])
			dist_pred = Beta(pred['future_mu'][i], pred['future_sigma'][i])
			kl_div = torch.distributions.kl_divergence(dist_sup, dist_pred)
			future_action_loss += torch.mean(kl_div[:, 0]) *0.5 + torch.mean(kl_div[:, 1]) *0.5
			future_feature_loss += F.mse_loss(pred['future_feature'][i], batch['future_feature'][i]) * self.config.features_weight
		future_feature_loss /= self.config.pred_len
		future_action_loss /= self.config.pred_len
		wp_loss = F.l1_loss(pred['pred_wp'], gt_waypoints, reduction='none').mean()
		loss = action_loss + speed_loss + value_loss + feature_loss + wp_loss+ future_feature_loss + future_action_loss
		self.log('train_action_loss', action_loss.item())
		self.log('train_wp_loss_loss', wp_loss.item())
		self.log('train_speed_loss', speed_loss.item())
		self.log('train_value_loss', value_loss.item())
		self.log('train_feature_loss', feature_loss.item())
		self.log('train_future_feature_loss', future_feature_loss.item())
		self.log('train_future_action_loss', future_action_loss.item())
		self.log('train_total_loss', loss.item())
		output = {
            "loss": loss,
            'train_action_loss':  action_loss,
			'train_speed_loss':  speed_loss,
			'train_value_loss': value_loss,
			'train_feature_loss': feature_loss,
			'train_wp_loss_loss': wp_loss,
			'train_future_feature_loss': future_feature_loss,
			'train_future_action_loss': future_action_loss,
			'train_loss': loss
        }
  
		return output

	def configure_optimizers(self):
		optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-7)
		lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 30, 0.5)
		return [optimizer], [lr_scheduler]

	def validation_step(self, batch, batch_idx):
		front_img = batch['front_img']
		speed = batch['speed'].to(dtype=torch.float32).view(-1,1) / 12.
		target_point = batch['target_point'].to(dtype=torch.float32)
		command = batch['target_command']
		state = torch.cat([speed, target_point, command], 1)
		value = batch['value'].view(-1,1)
		feature = batch['feature']
		gt_waypoints = batch['waypoints']

		pred = self.model(front_img, state, target_point)
		dist_sup = Beta(batch['action_mu'], batch['action_sigma'])
		dist_pred = Beta(pred['mu_branches'], pred['sigma_branches'])
		kl_div = torch.distributions.kl_divergence(dist_sup, dist_pred)
		action_loss = torch.mean(kl_div[:, 0]) *0.5 + torch.mean(kl_div[:, 1]) *0.5
		speed_loss = F.l1_loss(pred['pred_speed'], speed) * self.config.speed_weight
		value_loss = (F.mse_loss(pred['pred_value_traj'], value) + F.mse_loss(pred['pred_value_ctrl'], value)) * self.config.value_weight
		feature_loss = (F.mse_loss(pred['pred_features_traj'], feature) +F.mse_loss(pred['pred_features_ctrl'], feature))* self.config.features_weight
		wp_loss = F.l1_loss(pred['pred_wp'], gt_waypoints, reduction='none').mean()

		B = batch['action_mu'].shape[0]
		batch_steer_l1 = 0 
		batch_brake_l1 = 0
		batch_throttle_l1 = 0
		for i in range(B):
			throttle, steer, brake = self.model.get_action(pred['mu_branches'][i], pred['sigma_branches'][i])
			batch_throttle_l1 += torch.abs(throttle-batch['action'][i][0])
			batch_steer_l1 += torch.abs(steer-batch['action'][i][1])
			batch_brake_l1 += torch.abs(brake-batch['action'][i][2])

		batch_throttle_l1 /= B
		batch_steer_l1 /= B
		batch_brake_l1 /= B

		future_feature_loss = 0
		future_action_loss = 0
		for i in range(self.config.pred_len-1):
			dist_sup = Beta(batch['future_action_mu'][i], batch['future_action_sigma'][i])
			dist_pred = Beta(pred['future_mu'][i], pred['future_sigma'][i])
			kl_div = torch.distributions.kl_divergence(dist_sup, dist_pred)
			future_action_loss += torch.mean(kl_div[:, 0]) *0.5 + torch.mean(kl_div[:, 1]) *0.5
			future_feature_loss += F.mse_loss(pred['future_feature'][i], batch['future_feature'][i]) * self.config.features_weight
		future_feature_loss /= self.config.pred_len
		future_action_loss /= self.config.pred_len

		val_loss = wp_loss + batch_throttle_l1+5*batch_steer_l1+batch_brake_l1

		self.log("val_action_loss", action_loss.item(), sync_dist=True)
		self.log('val_speed_loss', speed_loss.item(), sync_dist=True)
		self.log('val_value_loss', value_loss.item(), sync_dist=True)
		self.log('val_feature_loss', feature_loss.item(), sync_dist=True)
		self.log('val_wp_loss_loss', wp_loss.item(), sync_dist=True)
		self.log('val_future_feature_loss', future_feature_loss.item(), sync_dist=True)
		self.log('val_future_action_loss', future_action_loss.item(), sync_dist=True)
		self.log('val_loss', val_loss.item(), sync_dist=True)
		return {'val_action_loss':  action_loss,
          		'val_speed_loss':  speed_loss,
				'val_value_loss': value_loss,
				'val_feature_loss': feature_loss,
				'val_wp_loss_loss': wp_loss,
				'val_future_feature_loss': future_feature_loss,
				'val_future_action_loss': future_action_loss,
				'val_loss': val_loss}

	def training_epoch_end(self, outputs):

		if(self.current_epoch == 1):
			# rand_img = torch.rand(1, 1, 28, 28) # TODO add the graph
			print(self.ref_front_img.shape)
			print(self.ref_front_img_all.shape)
			print(self.ref_front_img.unsqueeze(0).shape)
			# modelff = self.model(self.ref_front_img, self.ref_state, self.ref_target_point)
			# self.logger.experiment.add_graph(self.model, self.ref_front_img.unsqueeze(0), self.ref_state.unsqueeze(0), self.ref_target_point.unsqueeze(0))
			# logger._log_graph = True
   
		self.visActivations(self.ref_front_img, self.ref_state, self.ref_target_point)
        
		train_action_loss = torch.stack([x['train_action_loss'] for x in outputs]).mean()
		train_speed_loss =  torch.stack([x['train_speed_loss'] for x in outputs]).mean()
		train_value_loss = torch.stack([x['train_value_loss'] for x in outputs]).mean()
		train_feature_loss = torch.stack([x['train_feature_loss'] for x in outputs]).mean()
		train_wp_loss_loss = torch.stack([x['train_wp_loss_loss'] for x in outputs]).mean()
		train_future_feature_loss = torch.stack([x['train_future_feature_loss'] for x in outputs]).mean()
		train_future_action_loss = torch.stack([x['train_future_action_loss'] for x in outputs]).mean()
		train_loss = torch.stack([x['train_loss'] for x in outputs]).mean()
					
		self.logger[0].experiment.add_scalar("Loss/train_action_loss", train_action_loss, self.current_epoch)
		self.logger[0].experiment.add_scalar("Loss/train_speed_loss", train_speed_loss, self.current_epoch)
		self.logger[0].experiment.add_scalar("Loss/train_value_loss", train_value_loss, self.current_epoch)
		self.logger[0].experiment.add_scalar("Loss/train_feature_loss", train_feature_loss, self.current_epoch)
		self.logger[0].experiment.add_scalar("Loss/train_wp_loss_loss", train_wp_loss_loss, self.current_epoch)
		self.logger[0].experiment.add_scalar("Loss/train_future_feature_loss", train_future_feature_loss, self.current_epoch)
		self.logger[0].experiment.add_scalar("Loss/train_future_action_loss", train_future_action_loss, self.current_epoch)
		self.logger[0].experiment.add_scalar("Loss/train_loss", train_loss, self.current_epoch)
  
		self.histogram_adder()
		# return {'loss': train_loss}
	

	
	def histogram_adder(self, ):
		for name, params in self.named_parameters():
			self.logger[0].experiment.add_histogram(name, params, self.current_epoch) 
		return 

	def validation_epoch_end(self, outputs):
			val_action_loss = torch.stack([x['val_action_loss'] for x in outputs]).mean()
			val_speed_loss =  torch.stack([x['val_speed_loss'] for x in outputs]).mean()
			val_value_loss = torch.stack([x['val_value_loss'] for x in outputs]).mean()
			val_feature_loss = torch.stack([x['val_feature_loss'] for x in outputs]).mean()
			val_wp_loss_loss = torch.stack([x['val_wp_loss_loss'] for x in outputs]).mean()
			val_future_feature_loss = torch.stack([x['val_future_feature_loss'] for x in outputs]).mean()
			val_future_action_loss = torch.stack([x['val_future_action_loss'] for x in outputs]).mean()
			val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
						
			self.logger[0].experiment.add_scalar("Loss/val_action_loss", val_action_loss, self.current_epoch)
			self.logger[0].experiment.add_scalar("Loss/val_speed_loss", val_speed_loss, self.current_epoch)
			self.logger[0].experiment.add_scalar("Loss/val_value_loss", val_value_loss, self.current_epoch)
			self.logger[0].experiment.add_scalar("Loss/val_feature_loss", val_feature_loss, self.current_epoch)
			self.logger[0].experiment.add_scalar("Loss/val_wp_loss_loss", val_wp_loss_loss, self.current_epoch)
			self.logger[0].experiment.add_scalar("Loss/val_future_feature_loss", val_future_feature_loss, self.current_epoch)
			self.logger[0].experiment.add_scalar("Loss/val_future_action_loss", val_future_action_loss, self.current_epoch)
			self.logger[0].experiment.add_scalar("Loss/val_loss", val_loss, self.current_epoch)
			
			return {'val_loss': val_loss}

	def visActivations(self, img, state, target_point):
		
		# img = torch.Tensor.cpu(img).numpy().T
		# img = img.swapaxes(0, 1)
  		# print(img.shape)
		# image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		# pixels = np.array(image)
		# plt.imshow(img)
		# plt.show()
  
		self.logger[0].experiment.add_image("input_front_img", torch.Tensor.cpu(img), self.current_epoch, dataformats="CHW")
  
		img = img.unsqueeze(0)
		state = state.unsqueeze(0)
		target_point = target_point.unsqueeze(0)
  
		# print("vis in ---", img.shape, state.shape, target_point.shape)
		feature_emb, cnn_feature = self.model.perception(img)
		test_output = {}
		test_output['pred_speed'] = self.model.speed_branch(feature_emb)
		measurement_feature = self.model.measurements(state)
		
		j_traj = self.model.join_traj(torch.cat([feature_emb, measurement_feature], 1))
		test_output['pred_value_traj'] = self.model.value_branch_traj(j_traj)
		test_output['pred_features_traj'] = j_traj
		z = j_traj
		output_wp = list()
		traj_hidden_state = list()

		# initial input variable to GRU
		x = torch.zeros(size=(z.shape[0], 2), dtype=z.dtype).type_as(z)

		# autoregressive generation of output waypoints
		for _ in range(self.model.config.pred_len):
			x_in = torch.cat([x, target_point], dim=1)
			z = self.model.decoder_traj(x_in, z)
			traj_hidden_state.append(z)
			dx = self.model.output_traj(z)
			x = dx + x
			output_wp.append(x)

		pred_wp = torch.stack(output_wp, dim=1)
		test_output['pred_wp'] = pred_wp

		traj_hidden_state = torch.stack(traj_hidden_state, dim=1)
		init_att = self.model.init_att(measurement_feature).view(-1, 1, 8, 29)
		feature_emb = torch.sum(cnn_feature*init_att, dim=(2, 3))
		j_ctrl = self.model.join_ctrl(torch.cat([feature_emb, measurement_feature], 1))
		test_output['pred_value_ctrl'] = self.model.value_branch_ctrl(j_ctrl)
		test_output['pred_features_ctrl'] = j_ctrl
		policy = self.model.policy_head(j_ctrl)
		test_output['mu_branches'] = self.model.dist_mu(policy)
		test_output['sigma_branches'] = self.model.dist_sigma(policy)

		x = j_ctrl
		mu = test_output['mu_branches']
		sigma = test_output['sigma_branches']
		future_feature, future_mu, future_sigma = [], [], []

		# initial hidden variable to GRU
		h = torch.zeros(size=(x.shape[0], 256), dtype=x.dtype).type_as(x)

		for l in range(self.config.pred_len):
			x_in = torch.cat([x, mu, sigma], dim=1)
			h = self.model.decoder_ctrl(x_in, h)
			wp_att = self.model.wp_att(torch.cat([h, traj_hidden_state[:, _]], 1)).view(-1, 1, 8, 29)
			attention_map = torch.Tensor.cpu(wp_att.squeeze()).detach()
			self.logger[0].experiment.add_image("attention_map_" + str(l), attention_map, self.current_epoch, dataformats="HW")
			new_feature_emb = torch.sum(cnn_feature*wp_att, dim=(2, 3))
			merged_feature = self.model.merge(torch.cat([h, new_feature_emb], 1))
			dx = self.model.output_ctrl(merged_feature)
			x = dx + x
			policy = self.model.policy_head(x)
			mu = self.model.dist_mu(policy)
			sigma = self.model.dist_sigma(policy)
			future_feature.append(x)
			future_mu.append(mu)
			future_sigma.append(sigma)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	logger = TensorBoardLogger("tb_logs", name="original")
	wandb_logger = WandbLogger(name="original")

	parser.add_argument('--id', type=str, default='TCP', help='Unique experiment identifier.')
	parser.add_argument('--epochs', type=int, default=60, help='Number of train epochs.')
	parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate.')
	parser.add_argument('--val_every', type=int, default=3, help='Validation frequency (epochs).')
	parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
	parser.add_argument('--logdir', type=str, default='log', help='Directory to log data to.')
	parser.add_argument('--gpus', type=int, default=1, help='number of gpus')
	parser.add_argument('--loadfromcheckpoint', type=int, default=0, help='Load from n th checkpoint')
	parser.add_argument('--transferloading', type=bool, default=False, help='Load predefined part of the model.')


	args = parser.parse_args()
	args.logdir = os.path.join(args.logdir, args.id)

	# Config
	config = GlobalConfig()

	# Data
	train_set = CARLA_Data(root=config.root_dir_all, data_folders=config.train_data, img_aug = config.img_aug)
	print(len(train_set))
	val_set = CARLA_Data(root=config.root_dir_all, data_folders=config.val_data,)
	print(len(val_set))

	dataloader_train = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=8)
	dataloader_val = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=8)

	TCP_model = TCP_planner(config, args.lr)

	checkpoint_callback = ModelCheckpoint(save_weights_only=False, mode="min", monitor="val_loss", save_top_k=2, save_last=True,
											dirpath=args.logdir, filename="best_{epoch:02d}-{val_loss:.3f}")
	checkpoint_callback.CHECKPOINT_NAME_LAST = "{epoch}-last"
	trainer = pl.Trainer.from_argparse_args(args,
											default_root_dir=args.logdir,
											gpus = args.gpus,
											accelerator='ddp',
											sync_batchnorm=True,
											plugins=DDPPlugin(find_unused_parameters=False),
											profiler='simple',
											benchmark=True,
											log_every_n_steps=1,
											flush_logs_every_n_steps=5,
											callbacks=[checkpoint_callback,
														],
											check_val_every_n_epoch = args.val_every,
											max_epochs = args.epochs,
											logger=[logger, wandb_logger]
											)
	if args.loadfromcheckpoint>0:
		trainer = pl.Trainer(
			resume_from_checkpoint=args.logdir+"/epoch={checkpoint}-last.ckpt".format(checkpoint=args.loadfromcheckpoint),
			default_root_dir=args.logdir,
			gpus = args.gpus,
			accelerator='ddp',
			sync_batchnorm=True,
			plugins=DDPPlugin(find_unused_parameters=False),
			profiler='simple',
			benchmark=True,
			log_every_n_steps=1,
			flush_logs_every_n_steps=5,
			callbacks=[checkpoint_callback,
						],
			check_val_every_n_epoch = args.val_every,
			max_epochs = args.epochs,
			logger=[logger, wandb_logger])
		trainer.fit(TCP_model, dataloader_train, dataloader_val)
	elif args.transferloading:
		print("---------------------------------------------------------------------------")
		print("Transfer learning")
		lr = 0.0001
		config = GlobalConfig()
		planner = TCP_planner(config, lr=0.0001)
		model_dist = torch.load("/storage/scratch/e17-4yp-autonomous-driving/g04/TCPModels/best_model.ckpt", map_location=torch.device('cpu'))
		state_dict = model_dist["state_dict"]
		remove_prefix = 'model.'
		state_dict = {k[len(remove_prefix):] if k.startswith(remove_prefix) else k: v for k, v in state_dict.items()}
		state_dict1 = {}
		itr = 0
		for i in state_dict.keys():
			if "value_branch_traj" not in i:
				state_dict1[i]=state_dict[i]
			else:
				sizes = [torch.Size([512, 256]),
							torch.Size([512]),
							torch.Size([512, 512]),
							torch.Size([512]),
							torch.Size([1, 512]),
							torch.Size([1])]
				state_dict1[i] = torch.zeros(sizes[itr])
				itr+=1
		key_word = 'value_head'
		planner.model.load_state_dict(state_dict1)
		optimizer = optim.Adam(planner.model.parameters(), lr=lr, weight_decay=1e-7)
		lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 30, 0.5)
		optimizer_state_disk = model_dist["optimizer_states"][0]
		optimizer.load_state_dict(optimizer_state_disk)
		optimizer_schudular_disk = model_dist["lr_schedulers"][0]
		lr_scheduler.load_state_dict(optimizer_schudular_disk) 
		trainer.fit(planner, dataloader_train, dataloader_val)
	else:
		trainer.fit(TCP_model, dataloader_train, dataloader_val)
