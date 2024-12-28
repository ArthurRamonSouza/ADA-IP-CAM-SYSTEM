from core.centroidtracker import CentroidTracker
from core.trackableobject import TrackableObject

import cv2
import imutils
from imutils.video import VideoStream
from imutils.video import FPS

import time
import dlib
import argparse
import numpy as np

# Construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
help="path to Caffe 'deploy' prototxt file")

ap.add_argument("-m", "--model", required=True,
help="path to Caffe pre-trained model")

ap.add_argument("-i", "--input", type=str,
help="path to optional input video file")

ap.add_argument("-o", "--output", type=str,
help="path to optional output video file")

ap.add_argument("-c", "--confidence", type=float, default=0.4,
help="minimum probability to filter weak detections")

ap.add_argument("-s", "--skip-frames", type=int, default=30,
help="# of skip frames between detections")

args = vars(ap.parse_args())

print(f"Prototxt Path: {args['prototxt']}")
print(f"Model Path: {args['model']}")
print(f"Input Video Path: {args['input']}")
print(f"Output Video Path: {args['output']}")
print(f"Confidence Threshold: {args['confidence']}")
print(f"Skip Frames: {args['skip_frames']}")
