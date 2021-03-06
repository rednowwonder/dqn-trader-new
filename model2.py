#!/usr/bin/env python
# coding: utf-8

# In[15]:


import torch
import torch.nn as nn


# In[16]:


# input_dim = 20
# hidden_dim = 32
# num_layers = 2 
# output_dim = 2


# In[17]:


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.num_layers = num_layers

        # Building your LSTM
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()

        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()

        # One time step
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Index hidden state of last time step
        # out.size() --> 100, 28, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states! 
        out = self.fc(out[:, -1, :]) 
        # out.size() --> 100, 10
        return out


# In[18]:


# class LSTM(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers):
#         super(LSTM, self).__init__()
#         self.lstm = nn.LSTM(
#             input_size=input_size,
#             hidden_size=hidden_size,
#             num_layers=num_layers,
#         )

#         self.fc = nn.Linear(hidden_size)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, inputs):
#         out, (h_n, c_n) = self.lstm(inputs, None)
#         outputs = self.fc(h_n.squeeze(0))

#         return outputs


# In[ ]:




