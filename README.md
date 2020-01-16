# Pytorch Tutorial
Pytorch Tutorial for CS584 (Recommenders Systems) at Emory University

Please clone the repo if you want to run codes locally. 
Otherwise, if using Google Colab, download the repo and upload alexa toy data to your Google Drive to access inside Colab notebook (may need to modify data loading path, depending on which folder is set default for your Colab). You can copy the code to your Colab notebook to run.

The slides are also uploaded for reference.

### TODOs
1. Inside trainer_practice.py, try implementing the forward pass inside LSTM class. The output dimension should be (batch_size, output_size). If you are stuck, try looking at the code inside trainer_full.py

2. Once you are comfortable, try modifying the LSTM class to bidirectional or 2-layered architecture. These are additional parameter values for the LSTM module in PyTorch, but you need to modify the input dimensions of linear layers to accept new output from your LSTM. The output dimension should be identical.
