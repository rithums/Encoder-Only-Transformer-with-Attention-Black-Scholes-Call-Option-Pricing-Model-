import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler

from torch.utils.data import DataLoader
from dataloaders import finalDataset_series

from tqdm import tqdm
from torch.distributions.normal import Normal


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
        self.cdf = Normal(0, 1).cdf
        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(1)
        
    def black_scholes(self, v, other_input):
        v = torch.transpose(v, 0, 1)
        K, T, r, S = torch.transpose(other_input, 0, 1)
        S = torch.clamp(S, min=1e-5)    
        K = torch.clamp(K, min=1e-5)
        T = torch.clamp(T, min=1e-5)
        v = torch.clamp(v, min=1e-5)
        d1 = (torch.log(S/K) + (r + v**2/2)*T)/(v*torch.sqrt(T))
        d2 = d1 - v*torch.sqrt(T)
        P = self.cdf(d1)*S - self.cdf(d2)*K*torch.exp(-r*T)
        return torch.transpose(P, 0, 1)

    def forward(self, x, masks, target_input):
        embeddings = self.embeddingMLP.get_embeddings(x) #NxSxembeddingSize

        volatilities = self.encoderTransformation(embeddings, masks)
        volatilities = volatilities.view(-1, 1)

        volatilities.requires_grad_()
        price_input = torch.cat([volatilities, target_input],dim=1)

        prices = self.black_scholes(volatilities, target_input)

        return prices
    


embedding_size=16
model = architecture(embedding_size, 3, 2)

criterion = torch.nn.MSELoss(reduction="mean")
#optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
#scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[250,350], gamma=0.1) 


num_epochs = 500
batch_size = 256

seed = torch.Generator().manual_seed(20)

train_data, other = torch.utils.data.random_split(finalDataset_series(), [0.7,0.3], generator=seed)
val_data, test_data = torch.utils.data.random_split(other, [1/3, 2/3], generator=seed)

train_loader = DataLoader(dataset=train_data,batch_size=batch_size)
val_loader = DataLoader(dataset=val_data)
test_loader = DataLoader(dataset=test_data)

train_losses = []
val_losses = []

model_name = "fullModel_lr_5e-4_batch_256_blackScholes"

for e in range(num_epochs):

    model.train()

    for batch, batch_mask, target_input, target_price in tqdm(train_loader):

        optimizer.zero_grad()

        prices = model(batch, batch_mask, target_input)

        loss = criterion(prices, target_price)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
 

    with torch.no_grad():

        model.eval()
        #batch, batch_mask, target
        train_x, train_x_masks, train_y_input, train_y_price = train_data[:]
        train_pred = model(train_x, train_x_masks, train_y_input)
        train_loss = criterion(train_pred, train_y_price).item()
        train_losses.append(train_loss)

        val_x, val_x_masks, val_y_input, val_y_price = val_data[:]
        val_pred = model(val_x, val_x_masks, val_y_input)
        val_loss = criterion(val_pred, val_y_price).item()
        val_losses.append(val_loss)

        print("Epoch " + str(e) + ", Train Loss=" + str(train_loss) + ", Val Loss=" + str(val_loss))

    #scheduler.step()

    np.save("results/" + model_name + "_train_loss.npy", train_losses)
    np.save("results/" + model_name + "_val_loss.npy", val_losses)

    if e % 10 == 0:
        torch.save(model.state_dict(), "model_weights/" + model_name + ".pkl")
