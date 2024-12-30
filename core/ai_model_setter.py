import cv2

class ModelSetter:
    def __init__(self, model_path, prototxt_path):
        ''''
        Defines the Deep Neural Network, in Caffe format, thar will be used, the classes accepted by the model, 
        '''
        self.CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]
        
        # Caffe is a deep learning framework that lets you create and train neural networks and models. 
        # Like TensorFlow, PyTorch, and Keras, Caffe provides an abstraction over complex algorithms, 
        # allowing you to build models faster without worrying much about the underlying intricacies.
        self.net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)