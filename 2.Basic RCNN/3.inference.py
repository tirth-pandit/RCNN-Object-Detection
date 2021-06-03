import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import pickle
#from NMS import nms
from torchvision.ops import nms
import torch

argp = argparse.ArgumentParser()
argp.add_argument("-i", "--image", required=True,help="Image Path")
argp.add_argument("-m", "--model", required=True,help="Image Path")

args = vars(argp.parse_args())


## Read Image
img = cv2.imread( args["image"] )

# Selective Search 
print("Selective search...")
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
ss.setBaseImage(img)
ss.switchToSelectiveSearchFast()
rects = ss.process()

props = []
bbox = []

W,H,_ = img.shape

for (x, y, w, h) in rects[:300]:
	if w / float(W) < 0.1 or h / float(H) < 0.1:
            continue

	roi = img[y:y + h, x:x + w]
	roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
	roi = cv2.resize(roi, (150,150), interpolation=cv2.INTER_CUBIC)
	
	roi = img_to_array(roi)

	props.append(roi)
	bbox.append([x, y, x + w, y + h])

props = np.array(props, dtype="float32")
bbox = np.array(bbox, dtype="int32")

### Loading Model 
print("Loading model")
model = load_model(args["image"])

#print('Load Label Mapping')
#lm = pickle.loads(open('label_mapping', "rb").read())
lm = {0:'bird' , 1:'car' ,2:'neg'}

### Predicting Proposals
print('Predicting Proposals')
proba = model.predict(props)

labels = np.argmax(proba, axis=1)
idxs = np.where( labels != 2 )[0]
idxs = idxs.astype(int)


prob = []
labs = []
final_box = []

for i in list(idxs):
	pos = np.argmax(proba[i])
	probablity = proba[i][pos]
	
	if probablity >= 0.8:
		prob.append(probablity)
		labs.append(pos)
		final_box.append(bbox[i])

# final_box = np.array(final_box, dtype="int32")
#prob = np.array(prob)

img_copy = img.copy()

for (box ,p ,lab) in zip(final_box ,prob , labs) :
	# draw the bounding box, label, and probability on the image
	
	(startX, startY, endX, endY) = box
	cv2.rectangle( img_copy, (startX, startY), (endX, endY), (0, 255, 0), 2)
	
	y = startY - 10 if startY - 10 > 10 else startY + 10
	
	text_lab = lm[ lab ] 

	text=  text_lab + ' : '+ "{:.2f}%".format(p * 100)
	cv2.putText( img_copy, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

cv2.imwrite('Output/before_nms.jpg' ,img_copy )

final_b = torch.tensor(final_box ,dtype=torch.float32)
pr = torch.tensor(prob ,dtype=torch.float32)

boxIdxs = nms(final_b, pr ,iou_threshold=0.2)
for i in boxIdxs:
	# draw the bounding box, label, and probability on the image
	(startX, startY, endX, endY) = final_box[i]

	cv2.rectangle(img, (startX, startY), (endX, endY),(0, 255, 0), 2)
	
	y = startY - 10 if startY - 10 > 10 else startY + 10
	text= lm[lab] + ' : ' + "{:.2f}%".format(prob[i] * 100)
	cv2.putText(img, text, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
# show the output image *after* running NMS
cv2.imwrite("Output/after_nms.jpg", img)
