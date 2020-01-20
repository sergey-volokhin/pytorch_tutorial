from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import config, model, evaluate, data_utils
import os, time, argparse

if torch.cuda.is_available():
	device = 'cuda'
else:
	device = 'cpu'
print ('training on device:{}'.format(device.upper()))


############################## READ ARGS ##########################
parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--batch_size", type=int, default=16, help="batch size for training")
parser.add_argument("--epochs", type=int, default=15, help="training epoches")
parser.add_argument("--top_k", type=int, default=10, help="compute metrics@top_k")
parser.add_argument("--output_dim", type=int, default=32, help="predictive factors numbers in the model")
parser.add_argument("--num_ng", type=int, default=4, help="sample negative items for training")
args = parser.parse_args()


############################## PREPARE DATASET ##########################
train_data, test_data, user_num ,item_num, train_mat = data_utils.load_all()
train_dataset = data_utils.NCFData(train_data, item_num, train_mat, args.num_ng, True)
test_dataset = data_utils.NCFData(test_data, item_num, train_mat, 0, False)
train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
test_loader = data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)


########################### CREATE MODEL #################################
model = model.NCF(user_num, item_num, args.output_dim)
model = model.to(device)
loss_function = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)


########################### TRAINING #####################################
count, best_hr = 0, 0
for epoch in range(args.epochs):
	model.train()
	start_time = time.time()
	train_loader.dataset.ng_sample()

	for user, item, label in tqdm(train_loader, total=len(train_loader)):
		user = user.to(device)
		item = item.to(device)
		label = label.float().to(device)

		model.zero_grad()
		prediction = model(user, item)
		loss = loss_function(prediction, label)
		loss.backward()
		optimizer.step()
		count += 1
		break

	model.eval()
	HR, NDCG = evaluate.metrics(model, test_loader, args.top_k)

	elapsed_time = time.time() - start_time
	print("The time elapse of epoch {:03d}".format(epoch) + " is: " + time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))
	print("HR: {:.3f}\tNDCG: {:.3f}".format(np.mean(HR), np.mean(NDCG)))

	if HR > best_hr:
		best_hr, best_ndcg, best_epoch = HR, NDCG, epoch

print("End. Best epoch {:03d}: HR = {:.3f}, NDCG = {:.3f}".format(best_epoch, best_hr, best_ndcg))
