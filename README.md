# Object-Detection-Pipeline-Helper
 XML- Pascal voc to Text files/tf records,Testing frozen graph on images, Precision, Recall and mAP calculations, Confusion Matrix 
 
 
 
 
### For Converting Xml Coordinates to text files
Run the code by:
python convert.py directory_of_xml_files directory_of_text_files_where_coodinates_save

		>python convert.py GroundTruth_xml_files


### Coordinates Prediction Code:
Code use model file of previous code and Load images from images folder and run the model on those images and save the coordinates in Predicted_Results folder.

		>python coordinates_prediction.py images




### Evaluation:
The code will take Ground truth folder and 	Predicted_Results folder find the results of AP,Precision,Recall,total_TP,total_FP,mAP
To Run the code:

		>python evalutation.py --gt=GroundTruth_text_files --det=Predicted_Results --t=0.5

	# Where t is threshold value, that reperesent how much IOU(intersection over union we will consider it true.)
	
 
	
### Creating tf_records

The code will Create tf_records from xml files:

To Run the code:
		
		>python xml_to_tfrecord.py images GroundTruth_xml_files data/test.record
		
### Finding Confusion Matrix
The code will take test.record, model.pb,take label pbtxt and save confusion_matrix.csv
To Run the code:
		>python confusion_matrix.py data/test.record evaluator/ssd_mobilenet_v11_coco/frozen_inference_graph.pb evaluator/ssd_mobilenet_v11_coco/label.pbtxt confusion_matrix.csv
