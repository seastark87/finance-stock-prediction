import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split

# LSTM module
class lstm_model(pl.LightningModule) :
    def __init__(self, features, hiddens, seq_len, batch_size, num_layers, dropout, learning_rate, MSE) :
        super(lstm_model, self).__init__()
        # parameters
        self.features = features
        self.hiddens = hiddens
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.MSE = MSE 
        # Model Part
        self.lstm = nn.LSTM(input_size = features,
                            hidden_size = hiddens,
                            num_layers = num_layers,
                            dropout = dropout,
                            batch_first=True) 
        
        self.linear = nn.Sequential(
            nn.Linear(hiddens, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 6)
        )
        # profit logging
        self.invest_in = []

    def forward(self, x):
        x, _ = self.lstm(x) 
        output = self.linear(x[:,-1]) 
        return output
    

    def training_step(self, batch, batch_idx):  ## loss computation
        x, y = batch
        y_hat = self.forward(x)
        loss = self.MSE(y_hat, y)
        self.log('train_loss', loss, on_epoch=True)
        return loss


    def validation_step(self, batch, batch_idx):   ## 검증 
        x, y = batch
        y_hat = self(x)
        loss = self.MSE(y_hat, y)
        self.log('val_loss', loss)
    
    
    def test_step(self, batch, batch_idx):    ## test 데이터로더에서 제공하는 배치로 확인
        x, y = batch
        y_hat = self(x)
        loss = self.MSE(y_hat, y)
        self.log('test_loss', loss)
        for instance in range(len(y_hat)):
            self.invest_in.append(['prediction']+y_hat[instance, :].tolist()+['lable']+y[instance, :].tolist())


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)   ## adam optimizer
    
    
    # def training_epoch_end(self,outputs):
    #     #  the function is called after every epoch is completed

    #     # calculating average loss  
    #     avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

    #     # calculating correect and total predictions
    #     correct=sum([x["correct"] for  x in outputs])
    #     total=sum([x["total"] for  x in outputs])

    #     # logging using tensorboard logger
    #     self.logger.experiment.add_scalar("Loss/Train",
    #                                         avg_loss,
    #                                         self.current_epoch)
        
    #     self.logger.experiment.add_scalar("Accuracy/Train",
    #                                         correct/total,
    #                                         self.current_epoch)

    #     epoch_dictionary={
    #         # required
    #         'loss': avg_loss}

    #     return epoch_dictionary

    def test_step_end(self,outputs):
        #  the function is called after test step is completed

        # show the list of instance to invest in
        print(self.invest_in)
