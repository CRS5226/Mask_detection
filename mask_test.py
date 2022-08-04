# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
from datetime import datetime
import winsound
from plyer import notification
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

counter = 1
try:
	log = open("log.txt", "w")
except:
	print("not open log file")

def info():
	log.write(time.strftime("%c") + " not wearing mask" + "\n")

def sound():
	f1 = 5000
	duration = 500
	winsound.Beep(f1, duration)

def message():
	notification.notify(
		title="CAUTION",
		message="someone is not wearing mask"
	)

def news():
	print("no mask")

def img_capture():
	img_counter = 1
	img_name = "ss/opencv_frame_{}.jpg".format(img_counter)
	cv2.imwrite(img_name, frame)
	print("screenshot taken")
	# img_counter = img_counter + 1

def sent_mail():
	sender_email = "deepblueproject9@gmail.com"
	rec_email = "deepblueproject9@gmail.com"
	subject = "Mask wearing case"

	msg = MIMEMultipart()
	msg['From'] = sender_email
	msg['To'] = rec_email
	msg['Subject'] = subject

	body = "One visitor voilated Face mask policy. see in the camera to recognize user. A person has been detected without a face mask in the (address)"
	# message = "Subject:{}\n\n{}".format(subject, body)
	msg.attach(MIMEText(body, 'plain'))

	filename = 'ss/opencv_frame_1.jpg'
	attachment = open(filename, 'rb')

	part = MIMEBase('application', 'octet-stream')
	part.set_payload((attachment).read())
	encoders.encode_base64(part)
	part.add_header('Content-Disposition', "attachment; filename= " + filename)
	msg.attach(part)
	text = msg.as_string()

	server = smtplib.SMTP("smtp.gmail.com", 587)
	server.starttls()
	server.login(sender_email, "jee@61900")
	server.sendmail(sender_email, rec_email, text)
	print("message sent!!!!")
	server.quit()

def feature():
	news()
	info()
	img_capture()
	sent_mail()
	message()

def detect_and_predict_mask(frame, faceNet, maskNet):
	# grab the dimensions of the frame and then construct a blob
	# from it
	global numfaces
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()
	print(detections.shape)

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > 0.5:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# add the face and bounding boxes to their respective
			# lists
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	numfaces = len(faces)

	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	# return a 2-tuple of the face locations and their corresponding
	# locations
	return (locs, preds, numfaces)

# load our serialized face detector model from disk
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
maskNet = load_model("mask_detector.model")

# initialize the video stream
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
# vs = cv2.VideoCapture("maskvideo.mp4")

'''
vs = cv2.VideoCapture(0)
address = "http://192.168.1.102:8080/vs"y
vs.open(address)
'''

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	# ret, frame = vs.read()
	frame = vs.read()
	frame = imutils.resize(frame, width=700)

	# detect faces in the frame and determine if they are wearing a
	# face mask or not
	(locs, preds, numface) = detect_and_predict_mask(frame, faceNet, maskNet)

	# loop over the detected face locations and their corresponding
	# locations
	for (box, pred) in zip(locs, preds):
		# unpack the bounding box and predictions
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred

		# determine the class label and color we'll use to draw
		# the bounding box and text
		# label = "Mask" if mask > withoutMask else "No Mask"
		# color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

		if mask > withoutMask:
			label = "Mask"
			color = (0, 255, 0)
		else:
			label = "No Mask"
			color = (0, 0, 255)
			if counter == 1 and numface > 0:
				print("no of faces : ", numface)
				# feature()
				counter = 0

		# include the probability in the label
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
		# display the label and bounding box rectangle on the output
		# frame
		#cv2.putText(frame, "MADE BY CHITRAKSH", (0,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
		now = datetime.now()
		d_string = now.strftime("Date : %d/%m/%Y")
		t_string = now.strftime("Time : %I:%M:%S %p")
		cv2.rectangle(frame, (0,0), (250,65), color, -1)

		if mask > withoutMask:
			cv2.putText(frame, d_string, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
			cv2.putText(frame, t_string, (5, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
		else:
			cv2.putText(frame, d_string, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
			cv2.putText(frame, t_string, (5, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)


		cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	# show the output frame
	cv2.imshow("Mask Detection", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
log.flush()
log.close()
cv2.destroyAllWindows()
vs.stop()

# import feedback_form


