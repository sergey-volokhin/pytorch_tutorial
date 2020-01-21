# Day 2 - Movie ratings prediction using Neural Collaborative Filtering

### TODOs
1. Inside practice.py, try implementing class simpleCF. simpleCF receives a batch input of users and items. Convert two ID inputs into embedding vectors, feed to 3-layered MLP and directly predict ratings. The output of this model returns a tensor of predicted ratings, with shape (batch_input). If you achieve MAP <= 0.9, it is working.  


### Trained results of full NCF on rating prediction task
Epoch:1 -> test MAE:0.8815115094184875

Epoch:2 -> test MAE:0.7833456993103027

Epoch:3 -> test MAE:0.7571370005607605

Epoch:4 -> test MAE:0.741300106048584

Epoch:5 -> test MAE:0.733877420425415

Epoch:6 -> test MAE:0.7251555323600769

Epoch:7 -> test MAE:0.7211894392967224

Epoch:8 -> test MAE:0.7187884449958801

Epoch:9 -> test MAE:0.7180564999580383

Epoch:10 -> test MAE:0.7129073143005371


### Trained results of simpleCF (MLP module only) on rating prediction task
Epoch:1 -> test MAE:0.8548901677131653

Epoch:2 -> test MAE:0.8032435774803162

Epoch:3 -> test MAE:0.7829871773719788

Epoch:4 -> test MAE:0.7717518210411072

Epoch:5 -> test MAE:0.7604358196258545

Epoch:6 -> test MAE:0.7667009234428406

Epoch:7 -> test MAE:0.7586390972137451

Epoch:8 -> test MAE:0.7651193141937256

Epoch:9 -> test MAE:0.7613377571105957

Epoch:10 -> test MAE:0.756910502910614

