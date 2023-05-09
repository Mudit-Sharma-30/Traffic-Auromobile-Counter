import cv2
import numpy as np 

# starting the web camera
cap = cv2.VideoCapture('video.mp4')
min_width_rectangle = 80
min_height_rectangle = 80
count_line_position=550
Side_margin=630
    
#initialize substractor
algo = cv2.bgsegm.createBackgroundSubtractorMOG()

def center_handle(x,y,w,h):
    x1=int(w/2)
    y1=int(h/2)
    channelX=x+x1
    channelY=y+y1

    return channelX,channelY

detector =[]
offset =6 # allowable error in pixel
counter=0
Incoming=0
Outgoing=0

while(True):
    ret,frame1=cap.read()
    grey = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey,(3,3),5)
    
    # applying on its all frame
    img_sub = algo.apply(blur)
    dilat=cv2.dilate(img_sub,np.ones((5,5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    dilatada = cv2.morphologyEx(dilat,cv2.MORPH_CLOSE,kernel)
    dilatada = cv2.morphologyEx(dilatada,cv2.MORPH_CLOSE,kernel)
    counterShape,h = cv2.findContours(dilatada,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    cv2.line(frame1,(25,count_line_position),(1200,count_line_position),(255,127,0),3)
    #cv2.line(frame1,(Side_margin,count_line_position),(Side_margin,count_line_position+50),(255,127,0),3)

    for (i,c) in enumerate(counterShape):
        (x,y,weidth,height)=cv2.boundingRect(c)
        val_counter = (weidth>=min_width_rectangle) and (height>=min_height_rectangle)
        if not val_counter :
            continue

        cv2.rectangle(frame1, (x, y), (x + weidth, y + height), (0, 255, 0), 2)

        center = center_handle(x, y, weidth, height)
        detector.append(center)
        cv2.circle(frame1,center,4,(0,0,255),-1)

        for(x,y) in detector:
            if y<(count_line_position+offset) and y>(count_line_position-offset):
                counter+=1
                if x >Side_margin:
                    Outgoing +=1
                else :
                    Incoming +=1    
                cv2.line(frame1,(25,count_line_position),(1200,count_line_position),(127,255),3)
                detector.remove((x,y))
                # print("Vehicle Counter  : "+str(counter))
                # print("Vehicle Coming   : "+str(Incoming))
                # print("Vehicle Outgoing : "+str(Outgoing))
                

    cv2.putText(frame1,"VEHICLE COUNTER : "+str(counter),(  350,70),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),5)
    cv2.putText(frame1,"VEHICLE COMING : "+str(Incoming),(350,120),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
    cv2.putText(frame1,"VEHICLE OUTGOING : "+str(Outgoing),(750,120),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)

    cv2.imshow('Video Original',frame1)

    if cv2.waitKey(1)==13:
        break

cv2.destroyAllWindows()
cap.release




