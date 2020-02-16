from tqdm import tqdm
from sklearn.metrics import mean_absolute_error
import argparse
import numpy as np
import pandas as pd
from gensim import models

import torch
import torch.nn as nn
import torch.utils.data as data

from practice import simpleCF

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


'''
    i am using pre-trained Google News Word2Vec model for tags aggregating,
    so please download it and gunzip it (https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit)
    and specify the path to it in path_to_word2vec variable
'''
parser = argparse.ArgumentParser()
parser.add_argument('--path-to-word2vec', '-p', required=True)
parser.add_argument('--create-features', '-c', action='store_true')
args = parser.parse_args()

if args.create_features:
    path_to_word2vec = args.path_to_word2vec
    # path_to_word2vec = '/media/thejdxfh/Windows/Users/volok/Desktop/GoogleNews-vectors-negative300.bin'
    print('loading word2vec')
    w = models.KeyedVectors.load_word2vec_format(path_to_word2vec, binary=True)
    print('word2vec loaded')


'''
    for each tag get average embeddings for each word,
    multiply by the weight of that tag,
    and take the average of all tags
'''
def get_avg_embedding_for_movie(row):
    global groupped

    result = []
    try:
        for _, group in groupped.get_group(row['movieID']).iterrows():
            text = np.array([w[word] for word in group['value'].split() if word in w.vocab])
            if text.size > 0:
                result.append(text.mean(axis=0) * group['tagWeight'])
    except KeyError:
        return np.zeros(300)
    if len(result)<1:
        return np.zeros(300)
    return np.mean(result, axis=0)


def process_data(device, batch_size):
    global groupped

    datapath = '../data/'
    hetrec = datapath + 'hetrec2011-movielens-2k-v2/'

    if args.create_features:
        user_item_matrix = pd.read_csv(datapath + 'ml-latest-small/ratings.csv', usecols=[0, 1, 2]).rename(columns={'movieId': 'movieID'})
        movies = pd.read_csv(datapath + 'ml-latest-small/movies.csv', usecols=[0, 1])

        # get rt metadata
        movies_meta = pd.read_csv(hetrec + 'movies.dat', sep='\t', encoding='raw_unicode_escape', na_values='\N').rename(columns={'id': 'movieID'}).drop(['spanishTitle', 'imdbID', 'title', 'imdbPictureURL', 'rtID', 'rtAllCriticsNumReviews', 'rtTopCriticsNumReviews', 'rtPictureURL'], axis=1).fillna(0)
        movies_meta = movies_meta[movies_meta['movieID'].isin(user_item_matrix['movieID'].unique())]

        # get the average tags embeddings
        movie_tags = pd.read_csv(hetrec + 'movie_tags.dat', sep='\t')
        tags = pd.read_csv(hetrec + 'tags.dat', sep='\t', encoding='raw_unicode_escape').rename(columns={'id': 'tagID'})
        movie_tags = pd.merge(movie_tags, tags, on='tagID').sort_values(by=['movieID', 'tagID']).drop(['tagID'], axis=1)
        groupped = movie_tags.groupby('movieID')
        embedded_tags = pd.DataFrame.from_records(movies_meta.apply(get_avg_embedding_for_movie, axis=1), columns=[f'w2v_{i}' for i in range(1, 301)])
        movies_meta = pd.concat([movies_meta, embedded_tags], axis=1)

        user_item_matrix = pd.merge(user_item_matrix, movies_meta, on='movieID').sort_values(by=['userId', 'movieID'])

        # get a one-hot-encode-esque matrix of genres, then join on them
        movie_genres = pd.read_csv(hetrec + 'movie_genres.dat', sep='\t').pivot_table(index=['movieID'], columns=['genre'], aggfunc=[len], fill_value=0)
        movie_genres.columns = movie_genres.columns.droplevel(0)
        movie_genres = movie_genres.reset_index()
        user_item_matrix = pd.merge(user_item_matrix, movie_genres, on='movieID')

        # get a one-hot-encode matrix of countries, then join on them
        movie_countries = pd.get_dummies(pd.read_csv(hetrec + 'movie_countries.dat', sep='\t'))
        user_item_matrix = pd.merge(user_item_matrix, movie_countries, on='movieID').sort_values(by=['userId', 'movieID'])

        user_item_matrix.to_csv('user_item_matrix.tsv', sep='\t', index=False)
    else:
        user_item_matrix = pd.read_csv('user_item_matrix.tsv', sep='\t')

    give_test = lambda obj: obj.loc[np.random.choice(obj.index, len(obj.index) // 10), :]
    test_data = user_item_matrix.groupby('userId', as_index=False).apply(give_test).reset_index(level=0, drop=True)
    train_data = user_item_matrix[~user_item_matrix.index.isin(test_data.index)]

    # user item stats
    all_data = user_item_matrix
    num_user = len(all_data['userId'].unique()) + 1
    num_item = len(all_data['movieID'].unique()) + 1

    # convert input to torch tensors
    for _, columnData in train_data.iteritems():
        try:
            torch.tensor(columnData.values, device=device, dtype=torch.float)
        except:
            print(_)

    raise
    train_tensors = [torch.tensor(columnData.values, device=device, dtype=torch.float) for _, columnData in train_data.iteritems()]
    test_tensors = [torch.tensor(columnData.values, device=device, dtype=torch.float) for _, columnData in test_data.iteritems()]

    # convert tensors to dataloader
    train_dataset = data.TensorDataset(*train_tensors)
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = data.TensorDataset(*test_tensors)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print('train_samples:{} \t test_samples:{} \t num_user:{} \t num_item:{}'.format(train_data.shape[0], test_data.shape[0], num_user, num_item))
    return train_loader, test_loader, num_user, num_item


def train(lr, batch_size, output_dim=32):
    train_loader, test_loader, user_num, item_num = process_data(device=device, batch_size=batch_size)

    model = simpleCF(user_num, item_num, output_dim)

    model = model.to(device)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, 20):
        # training loop
        model.train()
        # for user, item, label in tqdm(train_loader, total=len(train_loader)):
        for features in tqdm(train_loader, total=len(train_loader)):
            label = features[2].float().to(device)
            model.zero_grad()
            prediction = model([feature.to(device) for feature in features[:2] + features[3:]])
            loss = loss_function(prediction, label)
            loss.backward()
            optimizer.step()

        # eval loop
        test_mae = []
        model.eval()
        # for user, item, label in test_loader:
        for features in test_loader:
            prediction = model([feature.to(device) for feature in features[:2] + features[3:]])
            label = features[2].float().detach().cpu().numpy()
            prediction = prediction.float().detach().cpu().numpy()
            test_mae.append(mean_absolute_error(y_pred=prediction, y_true=label))

        print('Epoch:{} -> test MAE:{}'.format(epoch, np.mean(test_mae)))


train(lr=0.0005, batch_size=256, output_dim=32)