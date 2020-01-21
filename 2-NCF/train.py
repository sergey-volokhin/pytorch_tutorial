from tqdm import tqdm
from sklearn.metrics import mean_absolute_error
from neuralCF import NCF
from practice import simpleCF

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data
import os, time

if torch.cuda.is_available():
	device = 'cuda'
else:
	device = 'cpu'
torch.manual_seed(10)
print('training on device:{}'.format(device.upper()))


# data processing function to read txt file and output pytorch dataloaders
def process_data(device, batch_size):
	train_rating = os.getcwd() + '/movie_train.txt'
	test_rating = os.getcwd() + '/movie_test.txt'
	train_data = pd.read_csv(train_rating, sep='\t', header=None, names=['user', 'item', 'rating'], usecols=[0, 1, 2])
	test_data = pd.read_csv(test_rating, sep='\t', header=None, names=['user', 'item', 'rating'], usecols=[0, 1, 2])

	# user item stats
	all_data = pd.concat([train_data, test_data])
	num_user = len(all_data['user'].unique()) + 1
	num_item = len(all_data['item'].unique()) + 1

	# convert input to torch tensors
	train_user = torch.tensor(train_data['user'].values, device=device)
	train_item = torch.tensor(train_data['item'].values, device=device)
	train_rating = torch.tensor(train_data['rating'].values, device=device, dtype=torch.float)

	test_user = torch.tensor(test_data['user'].values, device=device)
	test_item = torch.tensor(test_data['item'].values, device=device)
	test_rating = torch.tensor(test_data['rating'].values, device=device, dtype=torch.float)

	# convert tensors to dataloader
	train_dataset = data.TensorDataset(train_user, train_item, train_rating)
	train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	test_dataset = data.TensorDataset(test_user, test_item, test_rating)
	test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
	print ('train_samples:{} \t test_samples:{} \t num_user:{} \t num_item:{}'.format(len(train_rating), len(test_rating), num_user, num_item))
	return train_loader, test_loader, num_user, num_item


def train(lr, batch_size, output_dim=32, practice=True):
	train_loader, test_loader, user_num, item_num = process_data(device=device, batch_size=batch_size)

	if practice:
		model = simpleCF(user_num, item_num, output_dim)
	else:
		model = NCF(user_num, item_num, output_dim)

	model = model.to(device)
	loss_function = nn.MSELoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)

	for epoch in range(1, 20):
		# training loop
		start_time = time.time()
		model.train()
		for user, item, label in tqdm(train_loader, total=len(train_loader)):
			user = user.to(device)
			item = item.to(device)
			label = label.float().to(device)
			model.zero_grad()
			prediction = model(user, item)
			loss = loss_function(prediction, label)
			loss.backward()
			optimizer.step()

		# eval loop
		test_mae = []
		model.eval()
		for user, item, label in test_loader:
			user = user.to(device)
			item = item.to(device)
			prediction = model(user, item)
			label = label.float().detach().cpu().numpy()
			prediction = prediction.float().detach().cpu().numpy()
			MAE = mean_absolute_error(y_pred=prediction, y_true=label)
			test_mae.append(MAE)

		print ('Epoch:{} -> test MAE:{}'.format(epoch, np.mean(test_mae)))
		continue

# run this to test your simpleCF
train(lr=0.0005, batch_size=256, output_dim=32, practice=True)

# run this to compare with full neuralCF
train(lr=0.0005, batch_size=256, output_dim=32, practice=False)

