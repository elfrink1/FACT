from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch
import sys
import pandas as pd
import numpy as np


#torch.utils.data Dataset class
class Data(Dataset):
	def __init__(self, whole_data, labels=False, split=False):
		"""
			whole_data: features (numpy.array or torch.tensor)
			labels: labels for the data (numpy.array or torch.tensor)
			split: If True, split the dataset in training and validation sets
		"""
		self.data = whole_data
		self.labels = labels
		if split:
			self.data, self.valD, self.labels, self.valY = train_test_split(self.data, self.labels, test_size=0.25, random_state=42)
		
	@classmethod   
	def from_tsv(cls, path_features, path_labels=None, label_column_index=-1, split=False):
		"""
			Class method to initialize the Data class
			args:
			path_features: path to the data features file
			path_labels: path to the data labels file
			label_column_index: the index of the column which contains the labels in the features
								file if path_labels is None
			split: boolean Flag. Set to True if data is to be split in validation and training set

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
		#getitem at the corresponding index
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
		#length of the dataset
		return self.data.shape[0]