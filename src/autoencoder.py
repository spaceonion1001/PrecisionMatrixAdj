import torch
import torch.nn as nn

class AE(torch.nn.Module):
	def __init__(self, input_dim, hidden_dim=16):
		super().__init__()
		
		# Building an linear encoder with Linear
		# layer followed by Relu activation function
		# 784 ==> 9
		self.encoder = torch.nn.Sequential(
			torch.nn.Linear(input_dim, 64),
			torch.nn.ReLU(),
			torch.nn.Linear(64, 64),
			torch.nn.ReLU(),
			torch.nn.Linear(64, 32),
			torch.nn.ReLU(),
			torch.nn.Linear(32, hidden_dim)
		)
		
		# Building an linear decoder with Linear
		# layer followed by Relu activation function
		# The Sigmoid activation function
		# outputs the value between 0 and 1
		# 9 ==> 784
		self.decoder = torch.nn.Sequential(
			torch.nn.Linear(hidden_dim, 32),
			torch.nn.ReLU(),
			torch.nn.Linear(32, 64),
			torch.nn.ReLU(),
			torch.nn.Linear(64, 64),
			torch.nn.ReLU(),
			torch.nn.Linear(64, input_dim),
			#torch.nn.Sigmoid()
		)

	def forward(self, x):
		encoded = self.encoder(x)
		decoded = self.decoder(encoded)
		return decoded, encoded


def train_model(model, loss, optimizer, loader, epochs=20):
	for epoch in range(epochs):
		running_loss = 0
		for i, data in enumerate(loader):
			features, _ = data
			
			optimizer.zero_grad()
			
			outputs, _ = model(features)
			curr_loss = loss(outputs, features)
			curr_loss.backward()
			optimizer.step()
			running_loss += curr_loss.item()
		print("Epoch {} Loss {}".format(epoch, running_loss))
		
	return model