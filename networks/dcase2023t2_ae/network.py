import torch
from torch import nn

class AENet(nn.Module):
    def __init__(self,input_dim, block_size):
        super(AENet,self).__init__()
        self.input_dim = input_dim
        self.cov_source = nn.Parameter(torch.zeros(block_size, block_size), requires_grad=False)
        self.cov_target = nn.Parameter(torch.zeros(block_size, block_size), requires_grad=False)
        
        # jelteljesitmeny egysegnyire normalizalasa.
        # spectogramos tanitas kiprobalasa, frekvencia szamlalas es leosztas
        
        self.encoder = nn.Sequential(

            nn.Conv1d(1,32,kernel_size=3, padding='same'),
            nn.BatchNorm1d(32,momentum=0.01, eps=1e-03),
            nn.ReLU(),

            nn.Conv1d(32,32, kernel_size=3,  padding='same'),
            nn.BatchNorm1d(32,momentum=0.01, eps=1e-03),
            nn.ReLU(),

            nn.AvgPool1d(2),

            nn.Conv1d(32,64,kernel_size=3, padding='same'),
            nn.BatchNorm1d(64,momentum=0.01, eps=1e-03),
            nn.ReLU(),

            nn.Conv1d(64,64, kernel_size=3, padding='same'),
            nn.BatchNorm1d(64,momentum=0.01, eps=1e-03),
            nn.ReLU(),

            nn.AvgPool1d(2),

            nn.Conv1d(64,128, kernel_size=3, padding='same'),
            nn.BatchNorm1d(128,momentum=0.01, eps=1e-03),
            nn.ReLU(),

            nn.Conv1d(128,128, kernel_size=3,padding='same'),
            nn.BatchNorm1d(128,momentum=0.01, eps=1e-03),
            nn.ReLU(),

            nn.AvgPool1d(2),
            
        )

        self.decoder = nn.Sequential(
            
            nn.Conv1d(128,64, kernel_size=3,padding='same'),
            nn.BatchNorm1d(64,momentum=0.01, eps=1e-03),
            nn.ReLU(),

            nn.Conv1d(64,64, kernel_size=3,padding='same'),
            nn.BatchNorm1d(64,momentum=0.01, eps=1e-03),
            nn.ReLU(),

            nn.Upsample(scale_factor=2),

            nn.Conv1d(64,32,3, padding='same'),
            nn.BatchNorm1d(32,momentum=0.01, eps=1e-03),
            nn.ReLU(),

            nn.Conv1d(32,32,3, padding='same'),
            nn.BatchNorm1d(32,momentum=0.01, eps=1e-03),
            nn.ReLU(),

            nn.Upsample(scale_factor=2),

            nn.Conv1d(32,16, 3,padding='same'),
            nn.BatchNorm1d(16,momentum=0.01, eps=1e-03),
            nn.ReLU(),

            nn.Conv1d(16,16,3, padding='same'),
            nn.BatchNorm1d(16,momentum=0.01, eps=1e-03),
            nn.ReLU(),

            nn.Upsample(size=640),

            nn.Conv1d(16,1,3, padding='same'),

        )

    def forward(self, x):
              # + 1 channel
        z = self.encoder(x.view(-1,1,self.input_dim))
        #print("Encoder: ")
        #print(z.size())
        #print("Decoded")
        r = self.decoder(z)
        #print(r.size())
        return r.view(-1, self.input_dim), z
