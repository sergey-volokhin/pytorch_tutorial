import torch
import torch.nn as nn


class simpleCF(nn.Module):
    def __init__(self, user_num, item_num, output_dim):
        super(simpleCF, self).__init__()
        self.user_emb = nn.Embedding(user_num, output_dim * 4)
        self.item_emb = nn.Embedding(item_num, output_dim * 4)

        self.linear_1 = nn.Linear(output_dim * 8, output_dim * 4)
        self.linear_2 = nn.Linear(output_dim * 4, output_dim * 2)
        self.linear_3 = nn.Linear(output_dim * 2, 1)
        self.dropout = nn.Dropout()
        self.relu = nn.ReLU()

    def forward(self, user, item, *features):

        user_emb = self.user_emb(user)
        item_emb = self.item_emb(item)
        for i in features:
            print(features)
        concat = torch.cat((user_emb, item_emb, *features), -1)

        x = self.linear_1(concat)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.linear_2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.linear_3(x)
        x = x.view(-1)
        return x
