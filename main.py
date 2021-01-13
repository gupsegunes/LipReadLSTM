from utils import data_generator
import numpy as np
from tcn import compiled_tcn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model
from datetime import datetime
#import tensorflow as tf
#from tensorflow import keras
import keras
from keras.utils import to_categorical
from utils import wordCount,getFolderNamesInRootDir,wordArray
import os
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras import Sequential
from keras.layers import Dense,Conv2D, Flatten, MaxPooling2D, Dropout
from tcn import TCN
from keras.applications.vgg19 import VGG19
from keras.applications.vgg16 import VGG16
from keras_vggface.vggface import VGGFace
from keras.models import Model
from keras.layers.wrappers import TimeDistributed
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras import backend as K
from keras.applications.mobilenet import MobileNet
from keras import applications
from keras.optimizers import Nadam
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.recurrent import LSTM

x_train = None
y_train = None

x_val = None
y_val = None

x_test = None
y_test = None



y_train_one_hot_label = None
y_test_one_hot_label = None
y_val_one_hot_label = None


model = None

batch_size = 16 
now = datetime.now()
logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
class_names =[]

def create_class_names():
	for i in range(wordCount):
		class_names.append(wordArray[i])

def evaluate_model(model,x_test, y_test,x_val,y_val):
	print('Evaluating the model...')
	score = model.evaluate(x_val, y_val, batch_size=batch_size)
	print('Finished training, with the following val score:')
	print(score)
	print('Evaluating the model...')
	score = model.evaluate(x_test, y_test, batch_size=batch_size)
	print('Finished training, with the following val score:')
	print(score)

def create_save_plots(history,model,x_train, y_train, x_test,y_test, x_val,y_val,y_train_one_hot_label,y_test_one_hot_label,y_val_one_hot_label):
	create_plots(history)
	plot_and_save_cm(model,x_train, y_train, x_test,y_test, x_val,y_val,y_train_one_hot_label,y_test_one_hot_label,y_val_one_hot_label)

def plot_and_save_cm(model,x_train, y_train, x_test,y_test, x_val,y_val,y_train_one_hot_label,y_test_one_hot_label,y_val_one_hot_label):
	now = datetime.now()
	fileName = 'plots/conf_matrix_test_{0}.png'.format(now.strftime("%d_%m_%Y_%H%M%S"))
	y_pred = model.predict(x_test, verbose=1)
	y_pred_one_hot_label = y_pred.argmax(axis=1)
	plot_confusion_matrix(y_test_one_hot_label.squeeze(), y_pred_one_hot_label, classes=class_names,fileName=fileName)

	fileName = 'plots/conf_matrix_val_{0}.png'.format(now.strftime("%d_%m_%Y_%H%M%S"))
	y_pred = model.predict(x_val, verbose=1)
	y_pred_one_hot_label = y_pred.argmax(axis=1)
	plot_confusion_matrix(y_val_one_hot_label.squeeze(), y_pred_one_hot_label, classes=class_names,fileName=fileName)

def create_plots(history):
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


def plot_confusion_matrix(y_true, y_pred, classes,fileName,
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
	classes = np.array(classes)
	cm = confusion_matrix(y_true, y_pred)
	# Only use the labels that appear in the data
	classes = classes[unique_labels(y_true, y_pred)]
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
def ctc_loss_function(args):
	"""
	CTC loss function takes the values passed from the model returns the CTC loss using Keras Backend ctc_batch_cost function
	"""
	y_pred, y_true, input_length, label_length = args 
	# since the first couple outputs of the RNN tend to be garbage we need to discard them, found this from other CRNN approaches
	# I Tried by including these outputs but the results turned out to be very bad and got very low accuracies on prediction 
	y_pred = y_pred[:, 2:, :]
	return K.ctc_batch_cost(y_true, y_pred, input_length, label_length)

def create_bottleneck_model(x_train, y_train, x_test, y_test, x_val, y_val):
	np.random.seed(0)

	bottleneck_train_path = 'bottleneck_features_train_seen.npy'
	bottleneck_val_path = 'bottleneck_features_val_seen.npy'
	bottleneck_test_path = 'bottleneck_features_test_seen.npy'
	bottleneck_train_labels_path = 'bottleneck_features_train_labels_seen.npy'

	if not os.path.exists(bottleneck_train_path):
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
		bottleneck_model = Model(inputs=input_layer, outputs=x)
		if not os.path.exists(bottleneck_train_path):
			#bottleneck_features_train = bottleneck_model.predict_generator(self.training_generator(), steps=np.shape(self.X_train)[0] / self.config.batch_size)
			bottleneck_features_train = bottleneck_model.predict(x_train)
			np.save(bottleneck_train_path, bottleneck_features_train)
		if not os.path.exists(bottleneck_val_path):
			bottleneck_features_val = bottleneck_model.predict(x_val)
			np.save(bottleneck_val_path, bottleneck_features_val)
		if not os.path.exists(bottleneck_test_path):
			bottleneck_features_test = bottleneck_model.predict(x__test)
			np.save(bottleneck_test_path, bottleneck_features_test)
		if not os.path.exists(bottleneck_train_labels_path):
			np.save(bottleneck_train_labels_path, y_train)

		bottleneck_model.summary()
		plot_model(bottleneck_model, to_file='bottleneck_model_plot.png', show_shapes=True, show_layer_names=True)

def run_task():
	getFolderNamesInRootDir()
	create_class_names()

	print('loading saved data...')
	x_train = np.load('x_train.npy')
	y_train = np.load('y_train.npy')

	x_val = np.load('x_val.npy')
	y_val = np.load('y_val.npy')

	x_test = np.load('x_test.npy')
	y_test = np.load('y_test.npy')
	


	encoder = LabelBinarizer()
	y_train_one_hot_label = encoder.fit_transform(y_train)
	y_test_one_hot_label = encoder.fit_transform(y_test)
	y_val_one_hot_label = encoder.fit_transform(y_val)
	print(y_test_one_hot_label)
	'''
	y_train_one_hot_label = to_categorical(y_train, wordCount)
	y_test_one_hot_label = to_categorical(y_test, wordCount)
	y_val_one_hot_label = to_categorical(y_val, wordCount)
	'''

	y_train_one_hot_label = np.expand_dims(y_train_one_hot_label, axis=2)
	y_test_one_hot_label = np.expand_dims(y_test_one_hot_label, axis=2)
	y_val_one_hot_label = np.expand_dims(y_val_one_hot_label, axis=2)
	
	x_train = x_train.reshape(x_train.shape[0], 29,48,48,3)
	x_test = x_test.reshape(x_test.shape[0], 29,48,48,3)
	x_val = x_val.reshape(x_val.shape[0], 29,48,48,3)
	create_bottleneck_model(x_train, y_train, x_test, y_test, x_val, y_val)
	'''
	#TCN test
	#new model
	maxlen = 29
	maxsize = 2304
	max_features = 70000
	
	x_train = x_train.reshape(x_train.shape[0], x_train.shape[1])
	x_test = x_test.reshape(x_test.shape[0], x_test.shape[1])
	x_val = x_val.reshape(x_val.shape[0],x_val.shape[1])
	
	model = Sequential()
	model.add(Embedding(max_features, 4, input_shape=(maxlen*maxsize,)))
	model.summary()
	model.add(TCN(nb_filters=64,
				kernel_size=6,
				dilations=[1, 2, 4, 8, 16, 32, 64]))
	model.add(Dropout(0.5))
	model.add(Dense(wordCount, activation='softmax'))

	model.summary()

	# try using different optimizers and different optimizer configs
	model.compile('adam', 'sparse_categorical_crossentropy', metrics=['accuracy'])
	'''
	'''

	model = compiled_tcn(return_sequences=False,
							num_feat=1,
							num_classes=wordCount,
							nb_filters=20,
							kernel_size=5,
							dilations=[2 ** i for i in range(9)],
							nb_stacks=1,
							max_len=x_train[0:1].shape[1],
							use_skip_connections=False)
	
	print(f'x_train.shape = {x_train.shape}')
	print(f'y_train.shape = {y_train_one_hot_label.shape}')
	print(f'x_test.shape = {x_test.shape}')
	print(f'y_test.shape = {y_test_one_hot_label.shape}')

	model.summary()
	'''
	now = datetime.now()
	plot_file_name = 'plots/tcn_model_plot_' + now.strftime("%d_%m_%Y_%H%M%S") + '.png'
	plot_model(model, to_file=plot_file_name, show_shapes=True, show_layer_names=True)
	'''
	y_1 = y_train_one_hot_label.squeeze().argmax(axis=1)
	y_2 = y_test_one_hot_label.squeeze().argmax(axis=1)
	'''
	y_train_one_hot_label = y_train_one_hot_label.squeeze().argmax(axis=1)
	y_test_one_hot_label = y_test_one_hot_label.squeeze().argmax(axis=1)
	y_val_one_hot_label = y_val_one_hot_label.squeeze().argmax(axis=1)

	history = model.fit(x_train, y_train_one_hot_label, epochs=50,
				validation_data=(x_test, y_test_one_hot_label),callbacks=[tensorboard_callback])

	create_save_plots(history,model,x_train, y_train, x_test,y_test, x_val,y_val,y_train_one_hot_label,y_test_one_hot_label,y_val_one_hot_label)
	evaluate_model(model,x_test,y_test_one_hot_label,x_val,y_val_one_hot_label)



if __name__ == '__main__':
	run_task()