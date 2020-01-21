import torch
import torch.nn as nn
import torch.nn.functional as F

class simpleCF(nn.Module):
	def __init__(self, user_num, item_num, output_dim):
		super(simpleCF, self).__init__()
		# TODO: user_num: total num of users
		# TODO: item_num: total num of items
		# TODO: output_dim: number of predictive factors (neurons) on last MLP layer
        # TODO: initialize layers and activations here based on these dimensions

	def forward(self, user, item):
		# TODO: receive two inputs <- (batch_size, users), (batch_size, items)
        # TODO: output -> one output tensor of predicted ratings (batch_size)
		# TODO: implement forward pass based on figure 2, https://arxiv.org/pdf/1708.05031.pdf
		if torch.cuda.is_available():
			dummy_output = torch.ones(user.shape[0], device='cuda')
		else:
			dummy_output = torch.ones(user.shape[0], device='cpu')
		return dummy_output