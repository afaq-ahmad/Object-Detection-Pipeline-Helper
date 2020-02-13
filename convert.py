import _init_paths
import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import glob
import sys

def convert_annotation(in_file, out_file):
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    with open(out_file,'w') as f:
        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text

            xmlbox = obj.find('bndbox')
            xmin=int(xmlbox.find('xmin').text)
            ymin=int(xmlbox.find('ymin').text)
            xmax=int(xmlbox.find('xmax').text)
            ymax=int(xmlbox.find('ymax').text)

            f.write(' '.join([cls,str(xmin),str(ymin),str(xmax),str(ymax)])+'\n')

directory_load=sys.argv[1]

directory_save='GroundTruth_text_files'
if not os.path.exists(directory_save):
    os.makedirs(directory_save)

files=glob.glob(directory_load+'/*.xml')

for i in range(len(files)):
    xml_file_name=files[i].split('\\')[-1].split('/')[-1]
    txt_file_name=xml_file_name.replace('.xml','.txt')
    convert_annotation(directory_load+'/'+xml_file_name, directory_save+'/'+txt_file_name)
    
print('Xml coordinates are converted and saved in '+directory_save)