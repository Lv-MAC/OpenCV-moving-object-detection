import cv2 # for image
import time #delay
import imutils # resize

cam = cv2.VideoCapture(0) #camera hardware id, 0 for default, 1 for external camera/webcam
time.sleep(1)

firstFrame=None
area = 500

while True:
    _,img = cam.read() #read frame from camera
    text = "Normal"
    img = imutils.resize(img, width=500) #resize

    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #color 2 gray scale image

    gaussianImg = cv2.GaussianBlur(grayImg, (21,21), 0) #for smoothing image

    if firstFrame is None:
        firstFrame = gaussianImg #capturing first frame on first iteration
        continue

    imgDiff = cv2.absdiff(firstFrame, gaussianImg) #comparing plain frame with the blurred one to detect movement

    threshImg = cv2.threshold(imgDiff, 25, 255, cv2.THRESH_BINARY)[1] #convert to digital image

    threshImg = cv2.dilate(threshImg, None, iterations=2) #by using dilation, fixing pixels/holes in digital images

    cnts = cv2.findContours(threshImg.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #use contours to detect every object individually

    cnts = imutils.grab_contours(cnts)
    for c in cnts:
        if cv2.contourArea(c) < area:
            continue
        (x, y, w, h) = cv2.boundingRect(c) #object coordinates
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        text = "Moving Object Detected"
    print(text)
    cv2.putText(img, text, (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.imshow("cameraFeed",img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
cam.release()
cv2.destroyAllWindows()
