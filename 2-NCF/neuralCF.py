import torch
import torch.nn as nn
import torch.nn.functional as F 

# full neural collaborative filtering module
class NCF(nn.Module):
	def __init__(self, user_num, item_num, output_dim):
		super(NCF, self).__init__()
		self.GMF_layers = GMF(user_num=user_num, item_num=item_num, output_dim=output_dim)
		self.MLP_layers = MLP(user_num=user_num, item_num=item_num, output_dim=output_dim)
		self.predict_layer = nn.Linear(output_dim * 2, 1)
		self.dropout = nn.Dropout()

	def forward(self, user, item):
		# GMF layers
		output_GMF = self.GMF_layers(user, item)

		# MLP layers
		output_MLP = self.MLP_layers(user, item)

		# merge outputs
		concat = torch.cat((output_GMF, output_MLP), -1)
		concat = self.dropout(concat)
		prediction = self.predict_layer(concat).view(-1)
		return prediction


# multilayer perceptron sub-module
class MLP(nn.Module):
	def __init__(self, user_num, item_num, output_dim):
		super(MLP, self).__init__()
		self.user_emb = nn.Embedding(user_num, output_dim * 4)
		self.item_emb = nn.Embedding(item_num, output_dim * 4)

		self.linear_1 = nn.Linear(output_dim * 8, output_dim * 4)
		self.linear_2 = nn.Linear(output_dim * 4, output_dim * 2)
		self.linear_3 = nn.Linear(output_dim * 2, output_dim)
		self.dropout = nn.Dropout()
		self.relu = nn.ReLU()

	def forward(self, user, item):
		user_emb = self.user_emb(user)
		item_emb = self.item_emb(item)
		concat = torch.cat((user_emb, item_emb), -1)

		x = self.dropout(concat)
		x = self.linear_1(x)
		x = self.relu(x)

		x = self.dropout(x)
		x = self.linear_2(x)
		x = self.relu(x)

		x = self.dropout(x)
		x = self.linear_3(x)
		x = self.relu(x)
		return x

	def _init_weight_(self):
		nn.init.normal_(self.user_emb.weight, std=0.01)
		nn.init.normal_(self.item_emb.weight, std=0.01)


# generalized matrix factorization sub-module
class GMF(nn.Module):
	def __init__(self, user_num, item_num, output_dim):
		super(GMF, self).__init__()
		self.user_emb = nn.Embedding(user_num, output_dim)
		self.item_emb = nn.Embedding(item_num, output_dim)
		self._init_weight_()

	def forward(self, user, item):
		user_emb = self.user_emb(user)
		item_emb = self.item_emb(item)
		output = user_emb * item_emb
		return output

	def _init_weight_(self):
		nn.init.normal_(self.user_emb.weight, std=0.01)
		nn.init.normal_(self.item_emb.weight, std=0.01)
