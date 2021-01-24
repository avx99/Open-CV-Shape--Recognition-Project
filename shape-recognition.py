import cv2
import numpy as np

# Load the shape template or reference image
image = cv2.imread('someshapes.jpg')
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
ret ,thresh = cv2.threshold(gray,127,255,1)
contours ,hierarchy = cv2.findContours(thresh.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)


for c in contours:
    approx = cv2.approxPolyDP(c,0.01*cv2.arcLength(c,True),True)
    if len(approx) == 3:
        name = 'triangle'
        cv2.drawContours(image,[c],0,(0,255,0),-1)
        M = cv2.moments(c)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        cv2.putText(image,name,(cx,cy),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),1)

    elif len(approx) == 4:
        x,y,w,h = cv2.boundingRect(c)
        M = cv2.moments(c)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        if abs(w-h) <= 3:
            name = 'square'
            cv2.drawContours(image,[c],0,(0,155,155),-1)
            cv2.putText(image,name,(cx,cy),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),1)
        else:
            name = 'rectangle'
            cv2.drawContours(image,[c],0,(0,0,255),-1)
            M = cv2.moments(c)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            cv2.putText(image,name,(cx,cy),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),1)
    elif len(approx) == 10:
        name = 'star'
        cv2.drawContours(image,[c],0,(255,0,255),-1)
        M = cv2.moments(c)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        cv2.putText(image,name,(cx,cy),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),1)
    elif len(approx) >= 15:
        name = 'circle'
        cv2.drawContours(image,[c],0,(155,0,255),-1)
        M = cv2.moments(c)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        cv2.putText(image,name,(cx,cy),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),1)

cv2.imshow('shpaes recognition',image)
cv2.waitKey(0)







cv2.imshow('Thresh', thresh)
cv2.waitKey()
cv2.destroyAllWindows()