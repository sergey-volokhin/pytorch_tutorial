# Day 2 - Movie ratings prediction using Neural Collaborative Filtering

### TODOs
1. Inside practice.py, try implementing class simpleCF. simpleCF receives a batch input of users and items. Convert two ID inputs into embedding vectors, feed to 3-layered MLP and directly predict ratings. The output of this model returns a tensor of predicted ratings, with shape (batch_input). If you achieve MAP <= 0.9, it is working.  