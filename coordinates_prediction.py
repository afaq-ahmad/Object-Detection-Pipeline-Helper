import _init_paths
import sys
import cv2
import Object_parser
import glob
import os
import json
from keras.models import load_model

with open('evaluator/classes_names.json','r') as f:
    classes_name=json.load(f)



images_directory=sys.argv[1]
images_directory=images_directory+'/'
coordinates_save_directory='Predicted_Results'
coordinates_save_directory=coordinates_save_directory+'/'

if not os.path.exists(coordinates_save_directory):
    os.makedirs(coordinates_save_directory)

# Image_files=glob.glob(images_directory+'/*')
Image_files=os.listdir(images_directory)

model_path='evaluator/ssd_mobilenet_v11_coco/frozen_inference_graph.pb'
objparse=Object_parser.objectparser(model_path,threshold_confidence=0.5)



for PATH_TO_IMAGE in Image_files:
    PATH_TO_IMAGE=images_directory+PATH_TO_IMAGE
    text_file=open(PATH_TO_IMAGE.replace(images_directory,coordinates_save_directory).split('.')[0]+'.txt','w')
    image = cv2.imread(PATH_TO_IMAGE)
    All_Coordinates=objparse.object_data_from_image(image.copy())
    for k in range(len(All_Coordinates)):
        for p in range(len(All_Coordinates[k])):
            class_n,score,x1,y1,x2,y2=All_Coordinates[k][p]
            text_file.write(' '.join([classes_name[str(class_n)],str(score),str(x1),str(y1),str(x2),str(y2)])+'\n')
            
            # image=cv2.rectangle(image,(x1,y1),(x2,y2),(255,255,2),2)
            # cv2.imwrite(PATH_TO_IMAGE.replace('.jpg','_.jpg'),image)
            print(classes_name[str(class_n)],score,x1,y1,x2,y2)
            
print('Program Complete Results are saved in '+coordinates_save_directory)