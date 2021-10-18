import numpy as np
from tqdm import tqdm

PATH = '/itet-stor/wolflu/net_scratch/projects/EEGEyeNet_experimental/data/segmentation/Segmentation_stream_with_dots_synchronised_min.npz'
SAVE_PATH = '/itet-stor/wolflu/net_scratch/projects/EEGEyeNet_experimental/data/segmentation/Segmentation_stream_with_dots_synchronised_min.npz'

data = np.load(PATH)
X = data['EEG']
y = data['labels']

class Seg_Prep:
	def __init__(self, X, y) -> None:
		self.X = X
		self.y = y
		self.X_samples = []
		self.y_samples = []

	def prepare(self, X, y, sample_len):
		half_sample = int(sample_len/2)
		for i in tqdm(range(half_sample+1, len(X)-half_sample-1)):
			self.X_samples.append(X[i-half_sample:i+half_sample])
			self.y_samples.append(y[i])

	def save(self, path):
		np.savez(path, EEG=np.array(self.X), labels=np.array(self.y))
