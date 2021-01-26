from utils import data_generator
import numpy as np
from tcn import compiled_tcn
import matplotlib
matplotlib.use('Agg')
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model
from datetime import datetime

from keras.utils import to_categorical
from utils import wordCount,getFolderNamesInRootDir,wordArray,words
import os
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras import Sequential
from keras.layers import Dense,Conv2D, Flatten, MaxPooling2D, Dropout, Activation, TimeDistributed
from tcn import TCN
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg16 import VGG16
from keras_vggface.vggface import VGGFace
from tensorflow.keras.models import Model
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras import applications
from tensorflow.keras.optimizers import Nadam, Adam
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import LSTM, Bidirectional

class LipReading(object):

	def __init__(self):
		self.x_train = None
		self.y_train = None
		self.x_val = None
		self.y_val = None
		self.x_test = None
		self.y_test = None
		self.y_pred = None
		self.y_pred_one_hot_label = None
		self.y_train_one_hot_label = None
		self.y_test_one_hot_label = None
		self.y_val_one_hot_label = None
		self.model = None
		self.batch_size = 16 
		self.now = datetime.now()
		self.logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
		self.tensorboard_callback = keras.callbacks.TensorBoard(log_dir=self.logdir)
		self.class_names =[]
		self.bottleneck_train_path = 'bottleneck_features_train_seen.npy'
		self.bottleneck_val_path = 'bottleneck_features_val_seen.npy'
		self.bottleneck_test_path = 'bottleneck_features_test_seen.npy'
		self.bottleneck_train_labels_path = 'bottleneck_features_train_labels_seen.npy'
		self.bottleneck_model = None
		self.one_hot_labels_train = None
		self.one_hot_labels_test = None
		self.one_hot_labels_val = None
		self.iteration = 0
		self.train_data = None
		self.val_data = None
		self.test_data = None
		self.train_data_label= None
	

	def create_class_names(self):
		for i in range(wordCount):
			self.class_names.append(wordArray[i])

	def evaluate_model(self):
		print('Evaluating the model...')
		score = self.model.evaluate(self.x_val, self.y_val, batch_size=self.batch_size)
		print('Finished training, with the following val score:')
		print(score)
		print('Evaluating the model...')
		score = self.model.evaluate(self.x_test, self.y_test, batch_size=self.batch_size)
		print('Finished training, with the following val score:')
		print(score)

	def create_save_plots(self, history):
		create_plots(history)
		self.plot_and_save_cm(self)

	def plot_and_save_cm(self):
		encoder = LabelBinarizer()
		self.y_test_one_hot_label = encoder.fit_transform(self.y_test)
		self.y_val_one_hot_label = encoder.fit_transform(self.y_val)
		self.y_test_one_hot_label = np.expand_dims(self.y_test_one_hot_label, axis=2).squeeze().argmax(axis=1)
		self.y_val_one_hot_label = np.expand_dims(self.y_val_one_hot_label, axis=2).squeeze().argmax(axis=1)
		now = datetime.now()
		fileName = 'plots/conf_matrix_test_{0}.png'.format(now.strftime("%d_%m_%Y_%H%M%S"))
		self.y_pred = self.model.predict(self.x_test, verbose=1)
		self.y_pred_one_hot_label = self.y_pred.argmax(axis=1)
		self.plot_confusion_matrix(fileName=fileName)

		fileName = 'plots/conf_matrix_val_{0}.png'.format(now.strftime("%d_%m_%Y_%H%M%S"))
		self.y_pred = self.model.predict(self.x_val, verbose=1)
		self.y_pred_one_hot_label = self.y_pred.argmax(axis=1)
		self.plot_confusion_matrix(fileName=fileName)

	def create_plots(self,history):
		if not os.path.exists('plots'):
			os.mkdir('plots')

		#	print(history.history)
		now = datetime.now()
		# summarize history for accuracy
		print("create_plots {0}".format(now.strftime("%d_%m_%Y_%H%M%S")))
		plt.plot(history.history['accuracy'])
		plt.plot(history.history['val_accuracy'])
		plt.title('model accuracy')
		plt.ylabel('accuracy')
		plt.xlabel('epoch')
		plt.legend(['train', 'validation'], loc='upper left')

		fileName = 'plots/acc_plot_{0}.png'.format(now.strftime("%d_%m_%Y_%H%M%S"))


		plt.savefig(fileName)
		plt.clf()
		# summarize history for loss
		plt.plot(history.history['loss'])
		plt.plot(history.history['val_loss'])
		plt.title('model loss')
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.legend(['train', 'validation'], loc='upper left')

		fileName = 'plots/loss_plot_{0}.png'.format(now.strftime("%d_%m_%Y_%H%M%S"))


		plt.savefig(fileName)
		plt.clf()


	def plot_confusion_matrix(self,fileName,
							normalize=False,
							title=None,
							cmap=plt.cm.Blues,
							):
		"""
		This function prints and plots the confusion matrix.
		Normalization can be applied by setting `normalize=True`.
		"""
		if not title:
			if normalize:
				title = 'Normalized confusion matrix'
			else:
				title = 'Confusion matrix, without normalization'

		# Compute confusion matrix
		#classes =[0,1] 
		classes = np.array(self.class_names)
		cm = confusion_matrix(self.y_true, self.y_pred)
		# Only use the labels that appear in the data
		classes = classes[unique_labels(self.y_true, self.y_pred)]
		if normalize:
			cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
			print("Normalized confusion matrix")
		else:
			print('Confusion matrix, without normalization')

		print(cm)

		fig, ax = plt.subplots()
		im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
		ax.figure.colorbar(im, ax=ax)
		# We want to show all ticks...
		ax.set(xticks=np.arange(cm.shape[1]),
			yticks=np.arange(cm.shape[0]),
			# ... and label them with the respective list entries
			xticklabels=classes, yticklabels=classes,
			title=title,
			ylabel='True label',
			xlabel='Predicted label')

		# Rotate the tick labels and set their alignment.
		plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
				rotation_mode="anchor")

		# Loop over data dimensions and create text annotations.
		fmt = '.2f' if normalize else 'd'
		thresh = cm.max() / 2.
		for i in range(cm.shape[0]):
			for j in range(cm.shape[1]):
				ax.text(j, i, format(cm[i, j], fmt),
						ha="center", va="center",
						color="white" if cm[i, j] > thresh else "black")
		fig.tight_layout()
		
		plt.savefig(fileName)
		plt.clf()
		return ax

	def ctc_loss_function(self,args):
		"""
		CTC loss function takes the values passed from the model returns the CTC loss using Keras Backend ctc_batch_cost function
		"""
		y_pred, y_true, input_length, label_length = args 
		# since the first couple outputs of the RNN tend to be garbage we need to discard them, found this from other CRNN approaches
		# I Tried by including these outputs but the results turned out to be very bad and got very low accuracies on prediction 
		y_pred = y_pred[:, 2:, :]
		return K.ctc_batch_cost(y_true, y_pred, input_length, label_length)

	def create_bottleneck_model(self):
		np.random.seed(0)

		#train_generator = train_datagen.flow_from_directory(directory='train', class_mode='categorical', target_size=(64,64), batch_size=16, shuffle=True, classes=["dog", "cat"])

		#if not os.path.exists(bottleneck_train_path):
			#self.DataAugmentation()
		input_layer = keras.layers.Input(shape=(29, 48, 48, 3))
		# build the VGG16 network
		vgg_base = VGGFace(weights='vggface', include_top=False, input_shape=(48, 48, 3))
		#vgg_base = VGG16(weights='imagenet', include_top=False, input_shape=(self.config.MAX_WIDTH, self.config.MAX_HEIGHT, 3))
		vgg = Model(inputs=vgg_base.input, outputs=vgg_base.output)
		#vgg.trainable = False
		for layer in vgg.layers[:15]:
			layer.trainable = False
		x = TimeDistributed(vgg)(input_layer)
		self.bottleneck_model = Model(inputs=input_layer, outputs=x)
		if not os.path.exists(self.bottleneck_train_path):
			#bottleneck_features_train = bottleneck_model.predict_generator(self.training_generator(), steps=np.shape(self.X_train)[0] / self.config.batch_size)
			bottleneck_features_train = self.bottleneck_model.predict(self.x_train)
			np.save(self.bottleneck_train_path, bottleneck_features_train)
		if not os.path.exists(self.bottleneck_val_path):
			bottleneck_features_val = self.bottleneck_model.predict(self.x_val)
			np.save(self.bottleneck_val_path, bottleneck_features_val)
		if not os.path.exists(self.bottleneck_test_path):
			bottleneck_features_test = self.bottleneck_model.predict(self.x_test)
			np.save(self.bottleneck_test_path, bottleneck_features_test)
		if not os.path.exists(self.bottleneck_train_labels_path):
			np.save(self.bottleneck_train_labels_path, self.y_train)

		self.train_data = np.load(self.bottleneck_train_path)
		self.val_data = np.load(self.bottleneck_val_path)	
		self.test_data = np.load(self.bottleneck_test_path)
		self.train_data_label= np.load(self.bottleneck_train_labels_path)

		self.bottleneck_model.summary()
		plot_model(self.bottleneck_model, to_file='bottleneck_model_plot.png', show_shapes=True, show_layer_names=True)

	def create_model(wordCount, ne, msl, bs, lr, dp):


		np.random.seed(0)

		input_layer = keras.layers.Input(shape=(29, 48, 48, 3))
		# build the VGG16 network
		vgg_base = VGGFace(weights='vggface', include_top=False, input_shape=(48, 48, 3))
		#vgg_base = VGG16(weights='imagenet', include_top=False, input_shape=(self.config.MAX_WIDTH, self.config.MAX_HEIGHT, 3))
		vgg = Model(inputs=vgg_base.input, outputs=vgg_base.output)
		#vgg.trainable = False
		for layer in vgg.layers[:15]:
			layer.trainable = False
		x = TimeDistributed(vgg)(input_layer)

		self.model = Sequential(x)
		self.model.add(TimeDistributed(Flatten(),input_shape=self.train_data.shape[1:]))
		lstm1 = LSTM(32,return_sequences=True)
		lstm2 = LSTM(32,return_sequences=True)

		'''
		model.add(Bidirectional(lstm1, merge_mode='concat', weights=None))
		model.add(Bidirectional(lstm2))
		'''
		self.model.add(TCN(nb_filters=64,kernel_size=5,dilations=[1,2,4,8,16,32,64] ,use_batch_norm= True,use_skip_connections=True))
		#model.add(BatchNormalization())
		#model.add(BatchNormalization(momentum=0.99, epsilon=0.001))

		self.model.add(Dropout(rate=dp))
		self.model.add(Dense(wordCount))
		self.model.add(Activation('softmax'))
		adam = Adam(lr=lr)#, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
		self.model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

		print('Fitting the model...')
		self.class_names = np.array(words)
		self.iteration += 1

		fileName = 'csv/epoch_{0}.log'.format(self.iteration)

		csv_logger = keras.callbacks.CSVLogger(fileName, separator=',', append=True)
		#self.plot_confusion_matrix(self.y_test, y_pred,  title='Confusion matrix, without normalization')
		self.model.summary()
		plot_model(self.model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
		history = self.model.fit(self.train_data, self.one_hot_labels_train, epochs=ne, batch_size=bs,validation_data=(self.val_data, self.one_hot_labels_val),callbacks=[csv_logger])
		create_save_plots(history)
		evaluate_model(model,self.test_data, self.y_test_one_hot_label,self.val_data,self.y_val_one_hot_label)

	def run_task(self):
		getFolderNamesInRootDir()
		self.create_class_names()

		print('loading saved data...')
		self.x_train = np.load('x_train.npy')
		self.y_train = np.load('y_train.npy')

		self.x_val = np.load('x_val.npy')
		self.y_val = np.load('y_val.npy')

		self.x_test = np.load('x_test.npy')
		self.y_test = np.load('y_test.npy')
		

		
		encoder = LabelBinarizer()
		self.y_train_one_hot_label = encoder.fit_transform(self.y_train)
		self.y_test_one_hot_label = encoder.fit_transform(self.y_test)
		self.y_val_one_hot_label = encoder.fit_transform(self.y_val)
		print(self.y_test_one_hot_label)

		self.one_hot_labels_train = np.expand_dims(self.y_train_one_hot_label, axis=2).squeeze().argmax(axis=1)
		self.one_hot_labels_test = np.expand_dims(self.y_test_one_hot_label, axis=2).squeeze().argmax(axis=1)
		self.one_hot_labels_val = np.expand_dims(self.y_val_one_hot_label, axis=2).squeeze().argmax(axis=1)

		self.x_train = self.x_train.reshape(self.x_train.shape[0], 29,48,48,3)
		self.x_test = self.x_test.reshape(self.x_test.shape[0], 29,48,48,3)
		self.x_val = self.x_val.reshape(self.x_val.shape[0], 29,48,48,3)
		self.create_bottleneck_model()



		num_epochs = [40]#10
		learning_rates = [0.001, 0.002, 0.005]
		#learning_rates = [0.0005]
		batch_size = [64]
		dropout_ = [ 0.3, 0.5,0.7]
		#dropout_ = [ 0.4]
		self.iteration = 0
		for ne in num_epochs:
			for bs in batch_size: 
				for lr in learning_rates:
					for dp in dropout_:
						print("Epochs: {0} Batch Size:{1}  Learning Rate: {2} Dropout {3}".format(ne, bs, lr, dp))
						self.create_model (wordCount,ne,29, bs, lr, dp,train_data,test_data,val_data, y_train,y_test,y_val,y_train_one_hot_label,y_test_one_hot_label,y_val_one_hot_label,iteration)
	

if __name__ == '__main__':
	lr = LipReading()
	lr.run_task()