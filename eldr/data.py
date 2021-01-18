from torch.utils.data import Dataset, DataLoader
import torch


class Data(Dataset):
	def __init__(self, whole_data):
		self.data = whole_data
		
		
	def __getitem__(self, index):
		if not torch.is_tensor(self.data[index,:]):
			return torch.tensor(self.data[index,:]).view(1,-1)
		else:
			return self.data[index,:]
	
	def __len__(self):
		return self.data.shape[0]