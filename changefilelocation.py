import cv2
import os
import math
import argparse
from os.path import isdir 
from os import listdir
from os.path import isfile, join
from skimage.transform import resize   # for resizing images
import dlib 
from imutils import face_utils
import numpy as np
import dict2xml
from dicttoxml import dicttoxml
import zlib
import sys
RECTANGLE_LENGTH = 48
from pathlib import Path
import re
from sklearn import preprocessing
import copy
import logging
from shutil import copyfile

class lipReading(object):

	def __init__(self):
		#initial value of stepSize
		self.face_cascade = cv2.CascadeClassifier('../opencv/haarcascade_frontalface_default.xml')
		self.mouth_cascade=cv2.CascadeClassifier('/Users/gupsekobas/opencv_contrib-4.0.1/modules/face/data/cascades/haarcascade_mcs_mouth.xml')
		self.detector = dlib.get_frontal_face_detector()
		self.predictor = dlib.shape_predictor('../opencv/shape_predictor_68_face_landmarks.dat')
		self.xmouthpoints = []
		self.ymouthpoints = []
		self.image = None
		self.count = 0
		self.frame_dict =  {"Distance":{},"Angle":{}}
		self.frame_dict_norm =  {"Distance":{},"Angle":{}}
		#self.frame_dict =  {}
		#self.frame_dict_norm =  {}
		self.distanceArray = np.zeros(19)
		self.angleArray = np.zeros(19)
		self.distanceArrayNorm = np.zeros(19)
		self.angleArrayNorm = np.zeros(19)
		self.video_dict =  {}
		self.wordArray= []
		self.walk_dir = "../lip_reading/data"
		self.walk_dir_ = "../lip_reading/"
		self.datasets = ["train", "test","val"]
		self.fileNum = None
		self.targetDir = None
		(self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
		logging.basicConfig(handlers=[logging.FileHandler(filename="./log_records.txt", 
                                                 encoding='utf-8', mode='a+')],
                    format="%(asctime)s %(name)s:%(levelname)s:%(message)s", 
                    datefmt="%F %A %T", 
                    level=logging.ERROR)
		


	#this function calculate the difference of each point
	#in the mouth data to the 48th point.
	#mouth data has 20 points for every mouth detected.
	#it returns the normalized value of the distances in order that video resolution dont effect the output.
	def calculateDistanceOfMouthPoints(self):

		#normalizedx = preprocessing.normalize(np.array(self.xmouthpoints ).reshape(-1,1)) 
		norm = np.linalg.norm(self.xmouthpoints )
		normalizedx = self.xmouthpoints /norm
		norm = np.linalg.norm(self.ymouthpoints )
		normalizedy = self.ymouthpoints /norm
		for i in range(1 , 20):
			#find the distance between 2 points
			self.distanceArray[i-1] = math.hypot(self.xmouthpoints[i] - self.xmouthpoints[0] , self.ymouthpoints[i] - self.ymouthpoints[0])
			#find the angle between 2 points
			self.angleArray[i-1] = math.atan2(self.xmouthpoints[i] - self.xmouthpoints[0], self.ymouthpoints[i] - self.ymouthpoints[0])
			#find the distance between 2 points
			self.distanceArrayNorm[i-1] = math.hypot(normalizedx[i] - normalizedx[0] , normalizedy[i] - normalizedy[0])
			#find the angle between 2 points
			self.angleArrayNorm[i-1] = math.atan2(normalizedx[i] - normalizedx[0], normalizedy[i] - normalizedy[0])

		norm = np.linalg.norm(self.distanceArray)
		self.distanceArray = self.distanceArray/norm
		'''
		print(self.distanceArray)
		print(self.distanceArrayNorm)
		print(self.angleArray)
		print(self.angleArrayNorm)
		'''

		for i in range(1 , 20):
			'''
			distanceKey = "Distance_"+ str(i-1)
			angleKey = "Angle_"+ str(i-1)
			self.frame_dict[distanceKey] = {}
			self.frame_dict[angleKey] = {}
			self.frame_dict_norm[distanceKey] = {}
			self.frame_dict_norm[angleKey] = {}
			'''
			self.frame_dict['Distance'][str(i-1)]=self.distanceArray[i-1]
			self.frame_dict['Angle'][str(i-1)]= self.angleArray[i-1]
			self.frame_dict_norm['Distance'][str(i-1)]=self.distanceArrayNorm[i-1]
			self.frame_dict_norm['Angle'][str(i-1)]= self.angleArrayNorm[i-1]
			

	def get_mouth(self):
		#image = cv2.imread(filename)
		#image = imutils.resize(image, width=500)
		gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
		# detect faces in the grayscale image
		#rects = self.detector(gray, 1)
		rects = self.face_cascade.detectMultiScale(gray, scaleFactor = 1.2, minNeighbors = 5)
		if len(rects) == 1:
			rect = dlib.rectangle(rects[0][0],rects[0][1],rects[0][2],rects[0][3])

		if len(rects) > 1:
			rects = self.detector(gray, 1)
			if len(rects) < 1:
				logging.error( "ERROR: more than one face detected")
				return
			else:
				rect = rects[0]

		elif len(rects) < 1:
			rects = self.detector(gray, 1)
			if len(rects) < 1:
				logging.error( "ERROR: no faces detected")
				return
			else:
				rect = rects[0]




		shape = self.predictor(gray, rect)
		#shape = face_utils.shape_to_np(shape)
		#mouth = shape[lStart:lEnd]

		pad = 10
		self.xmouthpoints = [shape.part(x).x for x in range(self.lStart, self.lEnd)]
		self.ymouthpoints = [shape.part(x).y for x in range(self.lStart, self.lEnd)]
		#print(self.xmouthpoints)
		#print(self.ymouthpoints)
		self.calculateDistanceOfMouthPoints()
		
		maxx = max(self.xmouthpoints)
		minx = min(self.xmouthpoints)
		w = maxx - minx
		maxy = max(self.ymouthpoints)
		miny = min(self.ymouthpoints) 
		h = maxy- miny

		if w > h:
			rec_len = w
		else:
			rec_len = h

		w = rec_len +10
		h = rec_len +10
		minx= minx -5
		miny= miny -5
		'''
		for i in range(20):
			cv2.circle(self.image, (self.xmouthpoints[i],self.ymouthpoints[i]), radius=0, color=(0, 0, 255), thickness=2)
		'''
		crop_image = self.image[miny:miny + h,minx:minx+w]
		

		return crop_image

	def extract_mouth_data(self):
		


		mouth= self.get_mouth()
		if mouth is not None and mouth.size != 0 :
			#assuming there is only one face in the video
			#cv2.imwrite("frame%d.jpg" % self.count, self.image)     # save frame as JPEG file   
			mouth_filename =  self.targetDir  + "/"+ "mouth_{:05d}_{:05d}.jpg".format(self.fileNum ,self.count)
			#print(mouth.shape)
			mouth = cv2.resize(mouth, (RECTANGLE_LENGTH, RECTANGLE_LENGTH))
			#print(mouth.shape)
			cv2.imwrite(mouth_filename, mouth)
			print(mouth_filename, " saved")
			
			frame_key = "frame{:03d}".format( self.count)
			self.video_dict[frame_key] = {}
			self.video_dict[frame_key]= copy.deepcopy(self.frame_dict_norm)  
			self.count = self.count+1

		else:
			logging.error("no face written on file!")

	# Walk into directories in filesystem
	# Ripped from os module and slightly modified
	# for alphabetical sorting
	#
	def sortedWalk(self, top, topdown=True, onerror=None):
		from os.path import join, isdir, islink

		names = os.listdir(top)
		names.sort()
		dirs, nondirs = [], []

		for name in names:
			if isdir(os.path.join(top, name)):
				dirs.append(name)
			else:
				nondirs.append(name)

		if topdown:
			yield top, dirs, nondirs
		for name in dirs:
			path = join(top, name)
			if not os.path.islink(path):
				for x in self.sortedWalk(path, topdown, onerror):
					yield x
		if not topdown:
			yield top, dirs, nondirs

	def getFolderNamesInRootDir(self):
		

		print('walk_dir = ' + self.walk_dir)

		# If your current working directory may change during script execution, it's recommended to
		# immediately convert program arguments to an absolute path. Then the variable root below will
		# be an absolute path as well. Example:
		# walk_dir = os.path.abspath(walk_dir)
		print('walk_dir (absolute) = ' + os.path.abspath(self.walk_dir))

		for root, subdirs, files in self.sortedWalk(self.walk_dir):
			print('--\nroot = ' + root)
			for subdir in sorted(subdirs):
				print('\t- subdirectory ' + subdir)
				self.wordArray.append(subdir)
			break

	def createFoldersForEveryWord(self):
		for item in self.wordArray:
			
			Path("data_/train").mkdir(parents=True, exist_ok=True)
			Path("data_/test").mkdir(parents=True, exist_ok=True)
			Path("data_/val").mkdir(parents=True, exist_ok=True)
			Path("data_/test/"+item).mkdir(parents=True, exist_ok=True)
			Path("data_/train/"+item).mkdir(parents=True, exist_ok=True)
			Path("data_/val/"+item).mkdir(parents=True, exist_ok=True)

	def processVideos(self):
		print('walk_dir = ' + self.walk_dir)
		for item in self.wordArray:
		#	if item >= 'EVENTS':
			for subitem in self.datasets :
				sourceDir = self.walk_dir +"/" +item + "/" +subitem
				self.targetDir = "data_" +"/" +subitem + "/" +item
				for root, subdirs, files in self.sortedWalk(os.path.abspath(sourceDir)):
						for file in files:
							if file.endswith(".jpg"):
								filepath = os.path.join(root, file)
								targetfile = self.walk_dir_ + self.targetDir + '/' + file
								copyfile(filepath,targetfile)
								print("processing : ", filepath)



		# If your current working directory may change during script execution, it's recommended to
		# immediately convert program arguments to an absolute path. Then the variable root below will
		# be an absolute path as well. Example:
		# walk_dir = os.path.abspath(walk_dir)
		print('walk_dir (absolute) = ' + os.path.abspath(self.walk_dir))

		for root, subdirs, files in self.sortedWalk(self.walk_dir):
			print('--\nroot = ' + root)
			for subdir in sorted(subdirs):
				print('\t- subdirectory ' + subdir)
				self.wordArray.append(subdir)
			break
	def captureVideo(self, videoFileName,word):
		self.count = 0
		vidcap = cv2.VideoCapture(videoFileName)
		success,self.image = vidcap.read()
		if success == True:
			tmp_array = self.extract_mouth_data()
			while success:
				
				success,self.image = vidcap.read()
				if success == True:
					#print('Read a new frame: ', success)
					tmp_array = self.extract_mouth_data()

		xml =  dicttoxml(self.video_dict, custom_root='test', attr_type=False)
		#xmltodict()
		filename = "{}/d_{}_{:05d}.hgk".format(self.targetDir, word,self.fileNum)
		#print(xml)

		print ("size1: ", sys.getsizeof(xml))
		xmlz = zlib.compress(xml)
		print ("size2: ", sys.getsizeof(xmlz))
		f = open(filename, 'wb')
		f.write(xmlz)
		f.close()
		logging.error("Saved file " +filename )
def main():

	lr = lipReading()
	lr.getFolderNamesInRootDir()
	lr.createFoldersForEveryWord()
	lr.processVideos()
	distanceArray = []


if __name__ == '__main__':
	main()