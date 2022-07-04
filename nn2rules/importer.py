import tensorflow as tf
import pandas as pd
import numpy as np
import time

'''
Import model weights and data. The columns of the data are 
of the format featureName_featureVal. It loads the featureNames
and fixes an order of exploring the featureNames. 
'''
class Model:
	def __init__(self, model_path, data_path):
		self.model_path = model_path
		self.data_path = data_path

		self.__load_weights()
		self.__load_features()
		self.__reshape_feature_terms()

		self.__get_feature_order()
		return

	def __load_weights(self):
		model = tf.keras.models.load_model(self.model_path)
		self.layer_weights = []
		self.layer_bias = []

		for i in range(len(model.layers)):
			self.layer_weights.append(model.layers[i].get_weights()[0].T.tolist())
			self.layer_bias.append(model.layers[i].get_weights()[1].tolist())
		
		return

	def __load_features(self):
		self.feature_terms = list(pd.read_csv(self.data_path).columns[:-1])
		return

	def __reshape_feature_terms(self):
		feature_terms_root = [x.split('_')[0] for x in self.feature_terms]

		feature_names = []
		self.reshaped_feature_terms = []

		for i in range(len(feature_terms_root)):
			name = feature_terms_root[i]
			term = self.feature_terms[i]
			if(name not in feature_names):
				feature_names.append(name)
				self.reshaped_feature_terms.append([])
			self.reshaped_feature_terms[-1].append(term)
		return

	def __get_feature_order(self):
		feature_importance = []
		for i in range(len(self.layer_bias[0])):

			weights = self.layer_weights[0][i]
			bias = self.layer_bias[0][i]
			reshaped_feature_terms = [ft[:] for ft in self.reshaped_feature_terms]	
			
			reshaped_weights = []
			cum_sum = 0
			for ft in reshaped_feature_terms:
				num_terms = len(ft)
				fw = weights[cum_sum : cum_sum + num_terms]
				reshaped_weights.append(fw)
				cum_sum = cum_sum + num_terms

			feature_importance.append([])
			for i in range(len(reshaped_feature_terms)):
				feature_importance[-1].append( (max(reshaped_weights[i]) - min(reshaped_weights[i])) / len(reshaped_feature_terms[i]) )

		feature_importance = np.array(feature_importance)
		feature_importance = np.mean(feature_importance, axis = 0)

		self.feature_order = np.argsort(-feature_importance).tolist()
		return 

