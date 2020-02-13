import numpy as np
def non_max_suppression(boxes, overlapThresh=0.9):
	if len(boxes) == 0:
		return []
 
	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")
 
	# initialize the list of picked indexes	
	pick = []
 
	# grab the coordinates of the bounding boxes
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]
 
	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)
 
	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
 
		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])
 
		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)
 
		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]
 
		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))
 
	# return only the bounding boxes that were picked using the
	# integer data type
	return boxes[pick].astype("int")

def scores_update_nms(boxes,scores,width,height):
    #multiplying with width and height of image.
    boxes[:,0]=boxes[:,0]*height
    boxes[:,1]=boxes[:,1]*width
    boxes[:,2]=boxes[:,2]*height
    boxes[:,3]=boxes[:,3]*width
    boxes=boxes.astype('int')
    
    result_box_nms=non_max_suppression(boxes, overlapThresh=0.9)
    unq, count = np.unique(np.concatenate((boxes,result_box_nms)), axis=0, return_counts=True)
    removed_array=unq[count<2]
    #removed based on counting as of duplicates. if some is not duplicate then it will be act as removed box based on non_max_suppression.

    for k in [boxes.tolist().index(removed_array.tolist()[k]) for k in range(len(removed_array))]:
        scores[k]=0.0 # we are just replacing the score of overlap duplicate box to zero, so it will not be considered.
    return scores

