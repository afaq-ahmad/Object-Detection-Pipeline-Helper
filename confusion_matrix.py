# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Infers detections on a TFRecord of TFExamples given an inference graph.

Example usage:
  ./infer_detections \
    --input_tfrecord_paths=/path/to/input/tfrecord1,/path/to/input/tfrecord2 \
    --output_tfrecord_path_prefix=/path/to/output/detections.tfrecord \
    --inference_graph=/path/to/frozen_weights_inference_graph.pb

The output is a TFRecord of TFExamples. Each TFExample from the input is first
augmented with detections from the inference graph and then copied to the
output.

The input and output nodes of the inference graph are expected to have the same
types, shapes, and semantics, as the input and output nodes of graphs produced
by export_inference_graph.py, when run with --input_type=image_tensor.

The script can also discard the image pixels in the output. This greatly
reduces the output size and can potentially accelerate reading data in
subsequent processing steps that don't require the images (e.g. computing
metrics).
"""

import itertools
from utils import detection_inference
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import standard_fields
from utils import tf_example_parser
from utils import label_map_util
import itertools
import tensorflow as tf
from utils import detection_inference
import sys

input_tfrecord_paths=sys.argv[1] #'A comma separated list of paths to input TFRecords.'
inference_graph=sys.argv[2] #'Path to the inference graph with embedded weights.'
label_map=sys.argv[3] # 'Path to the label map''
discard_image_pixels=False #'Discards the images in the output TFExamples. This'
                        # ' significantly reduces the output size and is useful'
                        # ' if the subsequent tools don\'t need access to the'
                        # ' images (e.g. when computing evaluation measures).
                        
output_path=sys.argv[4]#'Path to the output the results in a csv.'

output_tfrecord_path='data/detections.tfrecord'
IOU_THRESHOLD = 0.5
CONFIDENCE_THRESHOLD = 0.5


def compute_iou(groundtruth_box, detection_box):
    g_ymin, g_xmin, g_ymax, g_xmax = tuple(groundtruth_box.tolist())
    d_ymin, d_xmin, d_ymax, d_xmax = tuple(detection_box.tolist())
    
    xa = max(g_xmin, d_xmin)
    ya = max(g_ymin, d_ymin)
    xb = min(g_xmax, d_xmax)
    yb = min(g_ymax, d_ymax)

    intersection = max(0, xb - xa + 1) * max(0, yb - ya + 1)

    boxAArea = (g_xmax - g_xmin + 1) * (g_ymax - g_ymin + 1)
    boxBArea = (d_xmax - d_xmin + 1) * (d_ymax - d_ymin + 1)

    return intersection / float(boxAArea + boxBArea - intersection)

def process_detections(detections_record, categories):
    record_iterator = tf.python_io.tf_record_iterator(path=detections_record)
    data_parser = tf_example_parser.TfExampleDetectionAndGTParser()

    confusion_matrix = np.zeros(shape=(len(categories) + 1, len(categories) + 1))

    image_index = 0
    for string_record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(string_record)
        decoded_dict = data_parser.parse(example)
        
        image_index += 1
        
        if decoded_dict:
            groundtruth_boxes = decoded_dict[standard_fields.InputDataFields.groundtruth_boxes]
            groundtruth_classes = decoded_dict[standard_fields.InputDataFields.groundtruth_classes]
            
            detection_scores = decoded_dict[standard_fields.DetectionResultFields.detection_scores]
            detection_classes = decoded_dict[standard_fields.DetectionResultFields.detection_classes][detection_scores >= CONFIDENCE_THRESHOLD]
            detection_boxes = decoded_dict[standard_fields.DetectionResultFields.detection_boxes][detection_scores >= CONFIDENCE_THRESHOLD]
            
            matches = []
            
            if image_index % 100 == 0:
                print("Processed %d images" %(image_index))
            
            for i in range(len(groundtruth_boxes)):
                for j in range(len(detection_boxes)):
                    iou = compute_iou(groundtruth_boxes[i], detection_boxes[j])
                    
                    if iou > IOU_THRESHOLD:
                        matches.append([i, j, iou])
                    
            matches = np.array(matches)
            if matches.shape[0] > 0:
                # Sort list of matches by descending IOU so we can remove duplicate detections
                # while keeping the highest IOU entry.
                matches = matches[matches[:, 2].argsort()[::-1][:len(matches)]]
                
                # Remove duplicate detections from the list.
                matches = matches[np.unique(matches[:,1], return_index=True)[1]]
                
                # Sort the list again by descending IOU. Removing duplicates doesn't preserve
                # our previous sort.
                matches = matches[matches[:, 2].argsort()[::-1][:len(matches)]]
                
                # Remove duplicate ground truths from the list.
                matches = matches[np.unique(matches[:,0], return_index=True)[1]]
                
            for i in range(len(groundtruth_boxes)):
                if matches.shape[0] > 0 and matches[matches[:,0] == i].shape[0] == 1:
                    confusion_matrix[groundtruth_classes[i] - 1][detection_classes[int(matches[matches[:,0] == i, 1][0])] - 1] += 1 
                else:
                    confusion_matrix[groundtruth_classes[i] - 1][confusion_matrix.shape[1] - 1] += 1
                    
            for i in range(len(detection_boxes)):
                if matches.shape[0] > 0 and matches[matches[:,1] == i].shape[0] == 0:
                    confusion_matrix[confusion_matrix.shape[0] - 1][detection_classes[i] - 1] += 1
        else:
            print("Skipped image %d" % (image_index))

    print("Processed %d images" % (image_index))

    return confusion_matrix

def display(confusion_matrix, categories, output_path):
    print("\nConfusion Matrix:")
    print(confusion_matrix, "\n")
    results = []

    for i in range(len(categories)):
        id = categories[i]["id"] - 1
        name = categories[i]["name"]
        
        total_target = np.sum(confusion_matrix[id,:])
        total_predicted = np.sum(confusion_matrix[:,id])
        
        precision = float(confusion_matrix[id, id] / total_predicted)
        recall = float(confusion_matrix[id, id] / total_target)
        
        #print('precision_{}@{}IOU: {:.2f}'.format(name, IOU_THRESHOLD, precision))
        #print('recall_{}@{}IOU: {:.2f}'.format(name, IOU_THRESHOLD, recall))
        
        results.append({'category' : name, 'precision_@{}IOU'.format(IOU_THRESHOLD) : precision, 'recall_@{}IOU'.format(IOU_THRESHOLD) : recall})
    
    df = pd.DataFrame(results)
    print(df)
    df.to_csv(output_path)

def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)
  global input_tfrecord_paths
  global label_map
  with tf.Session() as sess:
    input_tfrecord_paths = [
        v for v in input_tfrecord_paths.split(',') if v]
    tf.logging.info('Reading input from %d files', len(input_tfrecord_paths))
    serialized_example_tensor, image_tensor = detection_inference.build_input(
        input_tfrecord_paths)
    tf.logging.info('Reading graph and building model...')
    (detected_boxes_tensor, detected_scores_tensor,
     detected_labels_tensor) = detection_inference.build_inference_graph(
         image_tensor, inference_graph)

    tf.logging.info('Running inference and writing output to {}'.format(
        output_tfrecord_path))
    sess.run(tf.local_variables_initializer())
    tf.train.start_queue_runners()
    with tf.python_io.TFRecordWriter(
        output_tfrecord_path) as tf_record_writer:
      try:
        for counter in itertools.count():
          tf.logging.log_every_n(tf.logging.INFO, 'Processed %d images...', 10,
                                 counter)
          tf_example = detection_inference.infer_detections_and_add_to_example(
              serialized_example_tensor, detected_boxes_tensor,
              detected_scores_tensor, detected_labels_tensor,
              discard_image_pixels)
          tf_record_writer.write(tf_example.SerializeToString())
      except tf.errors.OutOfRangeError:
        tf.logging.info('Finished processing records')
    
    

    label_map = label_map_util.load_labelmap(label_map)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=100, use_display_name=True)

    confusion_matrix = process_detections(output_tfrecord_path, categories)

    display(confusion_matrix, categories, output_path)
    
    
if __name__ == '__main__':
    tf.app.run(main)