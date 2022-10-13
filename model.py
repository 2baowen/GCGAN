import torch
import torch.nn as nn
from GAT import GATSubNet,gcn
from Tcn import FCT

class Generator(nn.Module):
    def __init__(self, device, input_dim, gat_units, gatOut_dim, gat_heads, dropout, predict_length,
                 rnn_units, num_layers, alpha, tcn_units, d, kernel_size,node_nums):
        super(Generator, self).__init__()
        self.device = device
        self.predict_length = predict_length
        self.num_layers = num_layers
        self.hidden_size = rnn_units
        self.gcn = gcn(input_dim, gatOut_dim, node_nums)  #358/307/883/170
        #
        self.subnet = GATSubNet(input_dim, gat_units, gatOut_dim, gat_heads, alpha, dropout)
        self.gru = nn.GRU(gatOut_dim, rnn_units, num_layers, bidirectional=True)
        self.tcn = FCT(num_inputs=rnn_units *2, tcn_hidden=tcn_units, predict_length=predict_length, d=d,
                       kernel_size=kernel_size,dropout=dropout)

        self.linear_gra = nn.Linear(gatOut_dim,predict_length)
        self.linear_rnn = nn.Linear(rnn_units*2,predict_length)

    def forward(self, x, adj):
        gcn_output = self.gcn(x,adj[0])
        gat_output = self.subnet(x, adj[1])
        g = torch.sigmoid(gcn_output + gat_output)
        gat_output = g * gat_output + (1 - g) * gcn_output
        h0 = torch.randn(self.num_layers * 2, x.size(1), self.hidden_size).to(self.device)
        output ,h_n = self.gru(gat_output,h0)
        graph_out = self.linear_gra(gat_output).unsqueeze(-1)
        rnn_out = self.linear_rnn(output).unsqueeze(-1)
        out = self.tcn(output)
        out = graph_out + rnn_out + out
        return out,gcn_output,gat_output

class Discriminator(nn.Module):
    def __init__(self, num_nodes,):
        super(Discriminator, self).__init__()
        self.num_nodes = num_nodes
        self.model = nn.Sequential(
            nn.Linear(self.num_nodes, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.squeeze() # [B, (W+H), N]
        x_flat = x.reshape(-1, x.shape[2]) # [B*(W+H), N]

        validity = self.model(x_flat)

        return validity

'''
device='cuda'
x= torch.rand((64,883,12,1)).to(device)
y= torch.rand((64,883,12,1)).to(device)
g1 = torch.randn((883,883)).to(device)
g2 = torch.randn((883,883)).to(device)
g = [g1,g2]
num_nodes = 307
history_length = 12
predict_length = 12
train_ratio = 0.6
val_ratio = 0.2
test_ratio = 0.2
dropout=0.3
alpha=1
input_dim = 12
gat_units = 6
gat_heads = 4
gatOut_dim = 20
lstm_units = 64
num_layers = 2
tcn_units = 64
d = 9
kernel_size = 1
model = Generator(device,input_dim, gat_units, gatOut_dim, gat_heads, dropout, predict_length,
                 lstm_units, num_layers, alpha, tcn_units, d, kernel_size,883).to(device)
_,ss,qq = model(x,g)
print(_.shape)
'''