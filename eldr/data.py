from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch
import sys
import pandas as pd
import numpy as np


class Data(Dataset):
	def __init__(self, whole_data, labels=False, split=False):
		self.data = whole_data
		self.labels = labels
		if split:
			self.data, self.valD, self.labels, self.valY = train_test_split(self.data, self.labels, test_size=0.25, random_state=42)
		
	@classmethod   
	def from_tsv(cls, path_features, path_labels=None, label_column_index=-1, split=False):
		"""
			We use the dataset provided by the original repo
		"""
		if path_labels != None:
			X = pd.read_csv(path_features, sep="\t").to_numpy()
			y = pd.read_csv(path_labels, sep="\t").to_numpy()
		
			return cls(whole_data = X, labels=y, split=split)
		else:
			if label_column_index == None:
				sys.exit("Provide the label_column. Since you are providing the\
					path to the csv, we hope it contains the label_column.")
			else:
				d = pd.read_csv(path_features)
				X = d.iloc[:,:label_column_index].to_numpy()
				y = d.iloc[:,label_column_index].to_numpy()

			return cls(whole_data=X, labels=y, split=split)
	
	def __getitem__(self, index):
		if isinstance(self.labels, np.ndarray):
			if not torch.is_tensor(self.data[index,:]):
				return torch.tensor(self.data[index,:]).view(1,-1), torch.tensor(self.labels[index])
			else:
				return self.data[index,:], self.labels[labels]    
		else:
			if not torch.is_tensor(self.data[index,:]):
				return torch.tensor(self.data[index,:]).view(1,-1)
			else:
				return self.data[index,:]
			
			
	def __len__(self):
		return self.data.shape[0]