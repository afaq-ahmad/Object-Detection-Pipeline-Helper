from __future__ import division
from __future__ import print_function
from __future__ import absolute_import


import xml.etree.ElementTree as ET
import glob
import sys


import os
import io
import pandas as pd
import tensorflow as tf

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict
import json


image_dir=sys.argv[1]
Annotations_folder=sys.argv[2]
output_path=sys.argv[3]

Annotations=glob.glob(Annotations_folder+'/*.xml')

objects=[]
for i in range(len(Annotations)):
    image_name=Annotations[i].split('\\')[-1].replace('.xml','.jpg')

    tree = ET.parse(Annotations[i])
    root = tree.getroot()
    for member in root.findall('object'):
        bbx = member.find('bndbox')
        cls = member.find('name').text
        xmin = int(bbx.find('xmin').text)
        ymin = int(bbx.find('ymin').text)
        xmax = int(bbx.find('xmax').text)
        ymax = int(bbx.find('ymax').text)
        label = member.find('name').text

        objects.append(([image_name]+[cls]+[int(root.find('size')[0].text),int(root.find('size')[1].text)]+[xmin,ymin,xmax,ymax]))

Data_frame=pd.DataFrame(objects,columns=['filename','class','width','height','xmin','ymin','xmax','ymax'])


with open('Classes.txt','r') as f:
    Classes=json.load(f)

# TO-DO replace this with label map
def class_text_to_int(row_label):
    if row_label in Classes.keys():
        return Classes[row_label]
    else:
        None

def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


writer = tf.python_io.TFRecordWriter(output_path)
path = os.path.join(os.getcwd(), image_dir)

grouped = split(Data_frame, 'filename')
for group in grouped:
    tf_example = create_tf_example(group, path)
    writer.write(tf_example.SerializeToString())

writer.close()
output_path = os.path.join(os.getcwd(), output_path)
print('Successfully created the TFRecords: {}'.format(output_path))