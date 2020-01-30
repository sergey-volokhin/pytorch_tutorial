# Day 2 - Neural Collaborative Filtering (NCF)

## Model: Simple NCF
![Simplified NCF](NCF_simple.png)

## Model: Full NCF
![Simplified NCF](NCF_full.png)




## TODOs
1. Inside practice.py, try implementing class simpleCF. simpleCF receives a batch input of users and items. Convert two ID inputs into embedding vectors, feed to 3-layered MLP and directly predict ratings. The output of this model returns a tensor of predicted ratings, with shape (batch_input). If you achieve MAP <= 0.9, it is working. 


## Tutorial Task: Movie ratings prediction
| Epoch | MAE (full NCF) | MAE (simple CF) |
| :---: | :---: | :---: |
| 1 | 0.881 | 0.854 |
| 2 | 0.783 | 0.803 |
| 3 | 0.757 | 0.782 |
| 4 | 0.741 | 0.771 |
| 5 | 0.733 | 0.760 |
| 6 | 0.725 | 0.766 |
| 7 | 0.721 | 0.758 |
| 8 | 0.718 | 0.765 |
| 9 | 0.718 | 0.761 |
| 10 | 0.712 | 0.756 |


