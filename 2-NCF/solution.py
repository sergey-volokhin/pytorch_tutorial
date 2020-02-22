import torch
import torch.nn as nn


class simpleCF(nn.Module):
    def __init__(self, user_num, item_num, genre_num, country_num, tags_num, output_dim):
        super(simpleCF, self).__init__()
        self.user_emb = nn.Embedding(user_num, output_dim * 4)
        self.item_emb = nn.Embedding(item_num, output_dim * 4)
        self.genre_emb = nn.Embedding(genre_num, output_dim * 4)
        self.country_emb = nn.Embedding(country_num, output_dim * 4)
        self.tags_emb = nn.Embedding(tags_num, output_dim * 4)

        self.linear_1 = nn.Linear(output_dim * 8, output_dim * 4)
        self.linear_2 = nn.Linear(output_dim * 4, output_dim * 2)
        self.linear_3 = nn.Linear(output_dim * 2, 1)
        self.dropout = nn.Dropout()
        self.relu = nn.ReLU()

    def forward(self, user, item, genre, country, tags):

        user_emb = self.user_emb(user)
        item_emb = self.item_emb(item)
        genre_emb = self.genre_emb(genre)
        country_emb = self.country_emb(country)
        tags_emb = self.tags_emb(tags)
        for i in [user_emb, item_emb, genre_emb, country_emb, tags_emb]:
            print(i.shape)
        concat = torch.cat((user_emb, item_emb, genre_emb, country_emb, tags_emb), -1)

        x = self.linear_1(concat)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.linear_2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.linear_3(x)
        x = x.view(-1)
        return x
