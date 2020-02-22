from tqdm import tqdm
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import argparse
import numpy as np
import pandas as pd
from gensim import models

import torch
import torch.nn as nn
import torch.utils.data as data

from solution import simpleCF


'''
there are >95000 actors, >4000 directors, >5000 tags.
one-hot-encoding is not practical at this point, so we will not be using them as features.

'''

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
# torch.manual_seed(10)
print('training on device:{}'.format(device.upper()))


def process_data(device, batch_size):

    datapath = '../data/'
    hetrec = datapath + 'hetrec2011-movielens-2k-v2/'

    user_item_matrix = pd.read_csv(datapath + 'ml-latest-small/ratings.csv', usecols=[0, 1, 2]).rename(columns={'movieId':'movieID'})
    movies = pd.read_csv(datapath + 'ml-latest-small/movies.csv', usecols=[0, 1])

    movie_tags = pd.read_csv(hetrec + 'movie_tags.dat', sep='\t')
    movie_tags = movie_tags[movie_tags['movieID'].isin(user_item_matrix['movieID'])]
    new_dict = {a: b+1 for b, a in enumerate(sorted(movie_tags['tagID'].unique()))}
    movie_tags['tagID'] = movie_tags['tagID'].map(new_dict)

    # movie_tags['list'] = movie_tags.apply(lambda row: [row['tagID']] * row['tagWeight'], axis=1)
    movie_tags['list'] = movie_tags.apply(lambda row: [row['tagID']], axis=1)
    new_tags = movie_tags.groupby('movieID')['list'].sum()
    user_item_matrix['tags'] = user_item_matrix.apply(lambda row: new_tags.get(row['movieID']), axis=1)
    user_item_matrix = user_item_matrix.dropna()
    to_pad = max(user_item_matrix['tags'].apply(lambda value: len(value)))
    user_item_matrix['tags'] = user_item_matrix['tags'].apply(lambda value: np.array([0]*(to_pad-len(value))+value))

    movie_genres = pd.read_csv(hetrec + 'movie_genres.dat', sep='\t')
    genre_dict = {v: k+1 for k, v in enumerate(movie_genres['genre'].unique())}
    movie_genres['genre'] = movie_genres['genre'].replace(genre_dict)
    movie_genres['list'] = movie_genres.apply(lambda row: [row['genre']], axis=1)
    new_genres = movie_genres.groupby('movieID')['list'].sum()
    user_item_matrix['genres'] = user_item_matrix.apply(lambda row: new_genres.get(row['movieID']), axis=1)
    to_pad = max(user_item_matrix['genres'].apply(lambda value: len(value)))
    user_item_matrix['genres'] = user_item_matrix['genres'].apply(lambda value: np.array([0]*(to_pad-len(value))+value))

    countries = pd.read_csv(hetrec + 'movie_countries.dat', sep='\t')
    countries = countries[countries['movieID'].isin(user_item_matrix['movieID'])]
    user_item_matrix = pd.merge(user_item_matrix, countries, on='movieID').sort_values(by=['userId', 'movieID'])
    country_dict = {v: k+1 for k, v in enumerate(countries['country'].unique())}
    user_item_matrix["country"] = user_item_matrix["country"].map(country_dict)

    # remapping movieIDs so that torch stop yelling at me
    new_dict = {a: b+1 for b, a in enumerate(sorted(user_item_matrix['movieID'].unique()))}
    user_item_matrix['movieID'] = user_item_matrix['movieID'].map(new_dict)
    user_item_matrix = user_item_matrix.dropna()

    give_test = lambda obj: obj.loc[np.random.choice(obj.index, len(obj.index) // 10), :]
    test_data = user_item_matrix.groupby('userId', as_index=False).apply(give_test).reset_index(level=0, drop=True)
    train_data = user_item_matrix[~user_item_matrix.index.isin(test_data.index)]

    # user item stats
    # num_user = len(user_item_matrix['userId'].unique()) + 1  # 611
    # num_item = len(user_item_matrix['movieID'].unique()) + 1  # 5447
    # num_country = len(user_item_matrix['country'].unique()) + 1  # 56
    # num_genre = len(set(user_item_matrix['genres'].sum())) + 1  # 20
    # num_tags = len(set(user_item_matrix['tags'].sum())) + 1  # 5274
    # print(f"{num_user=}, {num_item=}, {num_country=}, {num_genre=}, {num_tags=}")
    num_user = user_item_matrix['userId'].nunique()
    num_item = user_item_matrix['movieID'].nunique()
    num_country = user_item_matrix['country'].nunique()
    num_genre = movie_genres['genre'].nunique() + 1
    num_tags = movie_tags['tagID'].nunique() + 1

    train_tensors = []
    test_tensors = []
    for tensors, dataset in [(train_tensors, train_data), (test_tensors, test_data)]:
        # convert input to torch tensors
        for column_name, columnData in dataset.iteritems():
            try:
                if column_name == 'rating':
                    tensors.append(torch.tensor(columnData.values, device=device, dtype=torch.float))
                else:
                    tensors.append(torch.tensor(columnData.values, device=device))
            except TypeError:
                new_array = [list(i) for i in columnData.values]
                tensors.append(torch.tensor(np.array(new_array), dtype=torch.long))

    # convert tensors to dataloader
    train_dataset = data.TensorDataset(*train_tensors)
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = data.TensorDataset(*test_tensors)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print('train_samples:{} \t test_samples:{} \t num_user:{} \t num_item:{}'.format(train_data.shape[0], test_data.shape[0], num_user, num_item))
    return train_loader, test_loader, num_user, num_item, num_genre, num_country, num_tags


def train(lr, batch_size, output_dim=32):
    train_loader, test_loader, user_num, item_num, genre_num, country_num, tags_num = process_data(device=device, batch_size=batch_size)

    model = simpleCF(user_num, item_num, genre_num, country_num, tags_num, output_dim)
    # def forward(self, user, item, genre, country, tags):

    model = model.to(device)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, 20):
        # training loop
        model.train()

        for user, item, label, tags, genre, country in tqdm(train_loader, total=len(train_loader)):
            user = user.to(device)
            item = item.to(device)
            genre = genre.to(device)
            country = country.to(device)
            tags = tags.to(device)

            label = label.float().to(device)
            model.zero_grad()

            prediction = model(user, item, genre, country, tags)
            loss = loss_function(prediction, label)
            loss.backward()
            optimizer.step()

        # eval loop
        test_mae = []
        model.eval()

        for user, item, label, genre, country in test_loader:
            user = user.to(device)
            item = item.to(device)
            genre = genre.to(device)
            country = country.to(device)

            prediction = model(user, item, genre, country)
            label = label.float().detach().cpu().numpy()
            prediction = prediction.float().detach().cpu().numpy()
            test_mae.append(mean_absolute_error(y_pred=prediction, y_true=label))

        print('Epoch:{} -> test MAE:{}'.format(epoch, np.mean(test_mae)))


train(lr=0.0005, batch_size=256, output_dim=32)
