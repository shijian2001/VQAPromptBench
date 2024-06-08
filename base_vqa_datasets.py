from typing import *

class SingleVQADatsetInstance:
	def get_standard_dataset(self):
		"(Abstract method) abstract VQA dataset method"

class BaseSingleVQADataset():
	def __init__(
		self,
		dataset_name: str
	):
		self.dataset_name = dataset_name
		self.dataset = None
		
	def get_dataset(self):
		return self.dataset.get_standard_dataset()

	
