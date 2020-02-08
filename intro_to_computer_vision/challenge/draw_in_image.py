import sys

sys.path.append('C:/users/patel/Lib/site-packages')

import cv2
import numpy as np
import matplotlib.pyplot as plt 
# True while mouse button down, False while mouse button up
# drawing = False
# ix = -1
# iy = -1

# def drawrectangle(event,x,y,flags,param):
#     global ix,iy,drawing
    
#     if event == cv2.EVENT_LBUTTONDOWN:
#         drawing == True
#         ix,iy = x,y
        
#     elif event == cv2.EVENT_MOUSEMOVE:
#         if drawing == TRUE:
#             cv2.rectangle( img,(ix,iy),(x,y),(0,255,0),-1)
            
#     elif event == cv2.EVENT_LBUTTONUP:
#         drawing == False
#         cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)
                           
                           
# def drawcircle(event,x,y,flags,param):
    
#     if event == cv2.EVENT_LBUTTONDOWN:
#         cv2.circle(img,(x,y),100,(0,255,0),-1) 
#     elif event == cv2.EVENT_RBUTTONDOWN:
#         cv2.circle(img,(x,y),100,(0,255,200),-1) 
        
# cv2.namedWindow(winname = 'mydrawing')

# cv2.setMouseCallback('mydrawing',drawrectangle)


# img = np.zeros((512,512,3))

# while True:
# cv2.imshow('mydrawing',img)

#     if cv2.waitKey(1) & 0xFF == 27:
#         break

# cv2.destroyAllWindows()