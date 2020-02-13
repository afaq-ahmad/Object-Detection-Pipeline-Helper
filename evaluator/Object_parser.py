import os,sys,glob,time
import cv2
import numpy as np
import tensorflow as tf
import os
from non_max_supperssion import scores_update_nms


class objectparser:
    """
    Class to detect object from an image.
    Parameters:
        model_path: Tensorflow Model Path
        threshold_confidence: (optional) Detected tables threshold confidence value. Default set at 0.5.
    """
    
    def __init__(self, model_path,threshold_confidence=0.5):    
        self.model_path=model_path
        self.threshold_confidence=threshold_confidence
        
        # Grab path to current working directory
        CWD_PATH = os.getcwd()
        # Path to frozen detection graph .pb file, which contains the model that is used
        # for object detection.
        PATH_TO_CKPT = os.path.join(CWD_PATH,model_path)

        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        # Load the Tensorflow model into memory.
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            sess = tf.Session(graph=detection_graph)

        # Input tensor is the image
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Output tensors are the detection boxes, scores, and classes
        # Each box represents a part of the image where a particular object was detected
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represents level of confidence for each of the objects.
        # The score is shown on the result image, together with the class label.
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        # Number of objects detected
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        
        self.detection_boxes=detection_boxes
        self.detection_scores=detection_scores
        self.detection_classes=detection_classes
        self.num_detections=num_detections
        self.image_tensor=image_tensor
        self.sess=sess

    def object_data_from_image(self, image):  
        
        """
        Function to detect object.
        
        Input Parameters:
        ----------------
            image: Numpy array of image in BGR(open-cv) format.
        
        Returns:
        --------
        All_Coordinates      list
        
        
        
        """
        
        image_expanded = np.expand_dims(image.copy(), axis=0)
        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_expanded})
        
        height,width,_=image.shape
        All_Coordinates=[]
        for i in range(len(boxes)):
            scores[i]=scores_update_nms(boxes[i].copy(),scores[i],width,height)        
        
        
            indexes_f=np.where(scores[i]>self.threshold_confidence)[0]
            coordinates_initial=[]
            for p in range(len(indexes_f)):
                y1,x1,y2,x2=boxes[i][[indexes_f[p]]][0]
                x1,y1,x2,y2=width*x1,height*y1,width*x2,height*y2
                score=scores[i][[indexes_f[p]]][0]
                cls_found=classes[i][[indexes_f[p]]][0]
                
                coordinates_initial.append([int(cls_found),score,int(x1),int(y1),int(x2),int(y2)])
                
            All_Coordinates.append(coordinates_initial)
        return All_Coordinates