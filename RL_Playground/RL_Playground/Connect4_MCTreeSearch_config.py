import os
import pickle

class Config:
	"""
	"""
	def __init__(self):
		### Self-Play
		CBversion = 'dirichlet'
		# Params for UCB from [3] (See maxUCB_edge below)
		self.PB_C_BASE = 19652
		self.PB_C_INIT = 1.25
		
		# Params for UCB from [4] (See maxUCB_edge below)
		self.ALPHA = 0.8
		self.EPS = 0.2
		self.C_PUCT = 1.0
		
		self.MEMORY_SIZE = 10


		### Model
		hparams = {'FILTER_NUM' : 256,
				   'KERNEL_SIZE' : '3__3',
				   'RELU_NEGATIVE_SLOPE' : 0.0,
				   'NUM_RESIDUAL_LAYERS' : 2,
				   'RESIDUAL_IN_FILTER_NUM' : 256,
				   'RESIDUAL_IN_KERNEL_SIZE' : '3__3',
				   'RESIDUAL_IN_RELU_NEGATIVE_SLOPE' : 0.0,
				   'RESIDUAL_OUT_RELU_NEGATIVE_SLOPE' : 0.0,
				   'POLICY_FILTER_NUM' : 2,
				   'POLICY_KERNEL_SIZE' : '1__1',
				   'POLICY_RELU_NEGATIVE_SLOPE' : 0.0,
				   'VALUE_FILTER_NUM' : 1,
				   'VALUE_KERNEL_SIZE' : '1__1',
				   'VALUE_IN_RELU_NEGATIVE_SLOPE' : 0.0,
				   'VALUE_IN_DENSE_UNITS' : 256,
				   'VALUE_OUT_RELU_NEGATIVE_SLOPE' : 0.0}






		### Training
		
		
		### Self-Play
		# self.num_actors = 5000
		# 
		# self.num_sampling_moves = 30
		# self.max_moves = 512  # for chess and shogi, 722 for Go.
		# self.num_simulations = 800
		# 
		# # Root prior exploration noise.
		# self.root_dirichlet_alpha = 0.3  # for chess, 0.03 for Go and 0.15 for shogi.
		# self.root_exploration_fraction = 0.25
		# 
		# # UCB formula
		# self.pb_c_base = 19652
		# self.pb_c_init = 1.25
		# 
		# ### Training
		# self.training_steps = int(700e3)
		# self.checkpoint_interval = int(1e3)
		# self.window_size = int(1e6)
		# self.batch_size = 4096
		# 
		# self.weight_decay = 1e-4
		# self.momentum = 0.9
		# # Schedule for chess and shogi, Go starts at 2e-2 immediately.
		# self.learning_rate_schedule = {
		#     0: 2e-1,
		#     100e3: 2e-2,
		#     300e3: 2e-3,
		#     500e3: 2e-4
		# }


        #### SELF PLAY
		EPISODES = 30
		MCTS_SIMS = 3
		MEMORY_SIZE = 30000
		TURNS_UNTIL_TAU0 = 10 # turn on which it starts playing deterministically
		CPUCT = 1
		EPSILON = 0.2
		ALPHA = 0.8
		
		
		#### RETRAINING
		BATCH_SIZE = 256
		EPOCHS = 1
		REG_CONST = 0.0001
		LEARNING_RATE = 0.1
		MOMENTUM = 0.9
		TRAINING_LOOPS = 10
		
		HIDDEN_CNN_LAYERS = [
			{'filters':75, 'kernel_size': (4,4)}
			 , {'filters':75, 'kernel_size': (4,4)}
			 , {'filters':75, 'kernel_size': (4,4)}
			 , {'filters':75, 'kernel_size': (4,4)}
			 , {'filters':75, 'kernel_size': (4,4)}
			 , {'filters':75, 'kernel_size': (4,4)}
			]
		
		#### EVALUATION
		EVAL_EPISODES = 20
		SCORING_THRESHOLD = 1.3


	def save(self, dir):
		with open(os.path.join(dir, 'config.pkl'), 'wb') as pklfile:
			pickle.dump(self.__dict__, pklfile)


	def _resolve_hparams(self, joint_text, dtype = int):
		return [dtype(x) for x in joint_text.split('__') if x ]
