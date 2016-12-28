import cv2
import numpy as np
import sys
import time

print 'Number of arguments:', len(sys.argv), 'arguments.'
if (len(sys.argv)==2):
    draw=1;
else:
    draw=0;

print(draw)
if draw==1:
    print('We will not detect hand instead track it')
    

hand = cv2.CascadeClassifier('hand.xml')
cap = cv2.VideoCapture(0)
found = 0
fourcc = cv2.cv.CV_FOURCC(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
 # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = (cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,10,1)
disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
count=50
while(cap.isOpened()):
    ret,img = cap.read()
    if found == 0 and draw == 0:
        #somethings to experiment are varying the scale and decreasing or increaasing the nieghbors
        hands = hand.detectMultiScale(img,1.05,100)
        for (x,y,w,h) in hands:
            if w!=0 and h!=0:
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,255),2)
                roi = img[y:y+int(round(h/2.0)),x:x+w]
                #roi = img[y:y+h,x:x+w]
                box = (x,y,w,h)
                hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
                hist = cv2.calcHist([hsv_roi],[0,1],mask,[180,256],[0,180,0,256])
                cv2.normalize(hist,hist,0,255,cv2.NORM_MINMAX)
                found = 1
    elif draw==1 or count>0:
        x=20
        y=20
        w=300
        h=300
        box = (x,y,w,h)
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,255),2)
        cv2.imwrite("hand.png",img)
        cv2.imshow("HandMarker",img)
        print('Please place your hand in the rectangular window')
        #time.sleep(1)#give 5 seconds for the user to put his hand in the window
        draw=0
        count=count-1
        roi = img[y:y+int(round(h/2.0)),x:x+w]
        hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
        hist = cv2.calcHist([hsv_roi],[0,1],mask,[180,256],[0,180,0,256])
        cv2.normalize(hist,hist,0,255,cv2.NORM_MINMAX)
        found=1
        
    else:
        bgr2hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([bgr2hsv],[0,1],hist,[0,180,0,256],1)    
        window, box = cv2.CamShift(dst, box, term_crit)
        (x,y,w,h) = box

        roi = img[y:y+h,x:x+w]
        groi = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
        _,out = cv2.threshold(groi,70,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        
        contours,_ = cv2.findContours(out,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        drawing = np.zeros(roi.shape,np.uint8)
        max_pts = 0
        for i in range(len(contours)):
            pts = len(contours[i])
            if(pts>=max_pts):
                max_pts = pts
                max_cnt = i
        print(len(contours))
        print(max_cnt)
        hand = contours[max_cnt]
        hull = cv2.convexHull(hand,returnPoints = False)
        defects = cv2.convexityDefects(hand,hull)
        cv2.drawContours(drawing,contours,max_cnt,(255,255,255),2)
        if defects is not None:
            for i in range(defects.shape[0]):
                s,e,f,d = defects[i,0]
                start = tuple(hand[s][0])
                end = tuple(hand[e][0])
                far = tuple(hand[f][0])
                cv2.circle(drawing,far,5,(255,0,0),-1)
                cv2.circle(drawing,start,5,(0,255,0),-1)
            hull = cv2.convexHull(hand)
            cv2.drawContours(drawing,[hull],0,(0,0,255),2)
        
        cv2.imshow('output',drawing)        
        cv2.ellipse(img,window, (0, 0, 255),0)
        #cv2.rectangle(img,track_window)

    #out.write(img)
    #mahotas.imsave('putput.mp4',img)
    cv2.imshow('',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
    
