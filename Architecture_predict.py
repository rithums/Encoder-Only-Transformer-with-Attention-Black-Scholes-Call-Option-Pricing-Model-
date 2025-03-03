import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler

from torch.utils.data import DataLoader
from dataloaders import finalDataset_series

from tqdm import tqdm


class EmbeddingMLP(nn.Module):
    def __init__(self, embedding_size):
        super(EmbeddingMLP, self).__init__()
        self.input_size = 5 
        self.embedding_size = embedding_size
        self.hidden_1 = nn.Linear(self.input_size, 128)
        self.hidden_2 = nn.Linear(128, embedding_size)

    def forward(self, x):
        out = self.hidden_1(x)
        out = F.relu(out)
        out = self.hidden_2(out)
        return out
    
    #Input size N x S x 5
    #N = batch size, S = number of data points in series
    def get_embeddings(self, X):
        N, S, _ = X.shape
        stacked = X.view(-1, self.input_size)
        out_stacked = self.forward(stacked)
        out = out_stacked.view(N, S, self.embedding_size)
        return out


class encoderOnlyTransformer(nn.Module):

    def __init__ (self,embedding_size, numLayers, n_heads):
        super(encoderOnlyTransformer, self).__init__()
        self.embedding_size = embedding_size

        self.encoderLayer = nn.TransformerEncoderLayer(
            d_model = embedding_size, nhead = n_heads, 
            dim_feedforward = 2048,dropout = 0.1
        )

        self.encoder = nn.TransformerEncoder(self.encoderLayer,
                                           num_layers = numLayers)#NxSxembedding size
        #out NxSxE
        #self.volatility = self.encoder[:, 0,0]
        

    def forward(self,embeddings, masks):
        masks = torch.transpose(masks, 0, 1)
        encoded = self.encoder(embeddings, src_key_padding_mask=masks)
        volatilities = encoded[:,-1,0]


        return volatilities
    
class PriceMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_size = 5
        self.hidden_1 = nn.Linear(self.input_size, 512)
        self.hidden_2 = nn.Linear(512, 256)
        self.hidden_3 = nn.Linear(256, 64)
        self.hidden_4 = nn.Linear(64, 1)

    def forward(self, x):
        out = self.hidden_1(x)
        out = F.relu(out)
        out = self.hidden_2(out)
        out = F.relu(out)
        out = self.hidden_3(out)
        out = F.relu(out)
        out = self.hidden_4(out)
        return out



class architecture(nn.Module):

    def __init__(self,embeddingSize, numLayers, nHeads):

        super(architecture, self).__init__()
        self.embeddingMLP = EmbeddingMLP(embeddingSize)

        self.encoderTransformation = encoderOnlyTransformer(embeddingSize,
                                                            numLayers, nHeads)
        
        self.priceMLP = PriceMLP()
  


    def forward(self, x, masks, target_input):
        embeddings = self.embeddingMLP.get_embeddings(x) #NxSxembeddingSize

        volatilities = self.encoderTransformation(embeddings, masks)
        volatilities = volatilities.view(-1, 1)

        #price_input = torch.cat([volatilities, target_input],dim=1)

        #target_input: Strike,  maturity, interestrate, stock price
        #volatility: volatilies

        #print(price_input.shape)
        #price_input = #concatenate target_input and volatilities
        #prices = self.black_scholes(volatilities, target_input) #self.priceMLP(price_input)
        #prices = self.priceMLP(price_input)
        prices = volatilities

        return prices
    


embedding_size=16
model = architecture(embedding_size, 3, 2)


seed = torch.Generator().manual_seed(20)

train_data, other = torch.utils.data.random_split(finalDataset_series(), [0.7,0.3], generator=seed)
val_data, test_data = torch.utils.data.random_split(other, [1/3, 2/3], generator=seed)


#model_name = "fullModel_lr_5e-4_batch_256_lrScheduler"
model_name = "fullModel"
model.load_state_dict(torch.load("model_weights/" + model_name + ".pkl"))


criterion = torch.nn.MSELoss(reduction="mean")


with torch.no_grad():

    model.eval()
    #batch, batch_mask, target
    train_x, train_x_masks, train_y_input, train_y_price = train_data[:]
    train_pred = model(train_x, train_x_masks, train_y_input)
    train_loss = criterion(train_pred, train_y_price).item()
    print(torch.stack([train_pred[:10],train_y_price[:10]]).T)
    print(train_loss)

    test_x, test_x_masks, test_y_input, test_y_price = test_data[:]
    test_pred = model(test_x, test_x_masks, test_y_input)
    test_loss = criterion(test_pred, test_y_price).item()
    print(torch.stack([test_pred[:10],test_y_price[:10]]).T)
    print(test_loss)

    #val_x, val_x_masks, val_y_input, val_y_price = val_data[:]
    #val_pred = model(val_x, val_x_masks, val_y_input)
    #val_loss = criterion(val_pred, val_y_price).item()


 