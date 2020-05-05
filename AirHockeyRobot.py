# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 09:49:14 2020

@author: Joshua Tanner
"""

import cv2
import numpy as np
import RPi.GPIO as GPIO
import time as tm

#---------------------------------------------------------------------------Jacob only edit here
ovsht = 6
factor = 0.3
confidence = 3
attack = 0.5

lower_pink = np.array([157,176,99]) #This is the puck
upper_pink = np.array([175,255,255])

lower_green = np.array([40,113,102])
upper_green = np.array([67,255,255])

record = True

#----------------------------------------------------------------------------Jacob

#
##-Setup Rpi
GPIO.setmode(GPIO.BCM)             # choose BCM or BOARD  
GPIO.setup(27, GPIO.OUT)       #Right
GPIO.setup(22, GPIO.OUT)       # left 
GPIO.setup(23, GPIO.OUT)       # Up 
GPIO.setup(24, GPIO.OUT)       # down


if attack > 1 or attack < 0:
    attack = 0.1


xl = 0
y1 = 0
t = []


def hm(stickx, sticky):
    
    if sticky - home[1] > ovsht:                  # move right
        if GPIO.input(22):
            GPIO.output(22, 0)
        if not GPIO.input(27):
            GPIO.output(27, 1)
#        print('go right')
    
    if sticky - home[1] < -ovsht :                 # move left
        if GPIO.input(27):
            GPIO.output(27, 0)
        if not GPIO.input(22):
            GPIO.output(22, 1)
#        print('left')
            
            
    if np.abs(sticky - home[1]) < ovsht :                 # dont move
        if GPIO.input(22):
            GPIO.output(22, 0)
        if GPIO.input(27):
            GPIO.output(27, 0)
#        print('stop')
        
    if stickx - home[0] > ovsht:                  # move up
        if GPIO.input(23):
            GPIO.output(23, 0)
        if not GPIO.input(24):
            GPIO.output(24, 1)
#        print('forward')
    
    if stickx - home[0] < -ovsht :                 # move down
        if GPIO.input(24):
            GPIO.output(24, 0)
        if not GPIO.input(23):
            GPIO.output(23, 1)
#        print('back')
            
            
    if np.abs(stickx - home[0]) < ovsht :                 # dont move
        if GPIO.input(24):
            GPIO.output(24, 0)
        if GPIO.input(23):
            GPIO.output(23, 0)
#        print('stop')
    return




#------------------------------------

def off():
        if GPIO.input(22):
            GPIO.output(22, 0)
        if GPIO.input(27):
            GPIO.output(27, 0)
        if GPIO.input(23):
            GPIO.output(23, 0)
        if GPIO.input(24):
            GPIO.output(24, 0)
#    print('stop')



trajx = []
trajy = []


def control(puckx, pucky, stickx, sticky, xavg):
    
        
    
    if pucky - sticky > ovsht:                  # move right
        if GPIO.input(27):
            GPIO.output(27, 0)
        if not GPIO.input(22):
            GPIO.output(22, 1)
#        print('go right')
    
    if pucky - sticky < -ovsht :                 # move left
        if GPIO.input(22):
            GPIO.output(22, 0)
        if not GPIO.input(27):
            GPIO.output(27, 1)
#        print('left')
            
            
    if np.abs(pucky - sticky) < ovsht :                 # dont move
        if GPIO.input(22):
            GPIO.output(22, 0)
        if GPIO.input(27):
            GPIO.output(27, 0)
#        print('stop')
    
    #102 94 104 20 
    #---------------------------------------------------- Attack
    if (stickx - xavg) > 0 and (stickx - xavg) < attack*np.abs(cropframewidth-smallframewidth):

        if stickx - xavg > ovsht:                  # move up
            if GPIO.input(23):
                GPIO.output(23, 0)
            if not GPIO.input(24):
                GPIO.output(24, 1)
        
    else:
        if GPIO.input(23):
            GPIO.output(23, 0)
        if GPIO.input(24):
            GPIO.output(24, 0) 

    
    return


def trajectory(xss, yss):
    
    
    delty = 0.0
    deltx = 0.0
   
    
    cnt2 = 1
    while cnt2 < len(xss):
        delty = delty + yss[cnt2] - yss[0]
        deltx = deltx + xss[cnt2] - xss[0]
        
        
        
        cnt2 = cnt2 + 1
        
           
    delty = delty/(len(xss)-1)
    deltx = deltx/(len(xss)-1)
    
    x = xss[len(xss)-1]
    y = yss[len(xss)-1]
    
    
    while (x <= cropframewidth) and (x >= smallframewidth) and (y <= frameheight) and (y >= 0):
        
        x = x + deltx
        y = y + delty
    
    x = int(x)
    y = int(y)
    
    if x > cropframewidth:
        x = cropframewidth
    if x < smallframewidth:
        x = smallframewidth
    if y > frameheight:
        y = frameheight
    if y < 0:
        y = 0
    
    valuex = int(x)
    valuey = int(y)
    
    
    return(valuex, valuey)



cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

cap.set(3,480)

cap.set(4,640)

cap.set(cv2.CAP_PROP_FPS, 120)


fps = cap.get(5)


if record:
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 30.0, (int(factor*cap.get(3)), int(factor*cap.get(4))))


framewidth  = int(cap.get(3))
frameheight = int(cap.get(4)) 


print(framewidth, frameheight, fps)

ret,frame = cap.read()

#frame = cv2.resize(frame, (0,0), fx = factor, fy=factor)

print(frame.shape)

r = cv2.selectROI('frame', frame)

r = [int(round(element*factor,0)) for element in r] 

cropframewidth = r[0] + r[2]
cropframeheight = r[1] + r[3]
smallframewidth = r[0]
smallframeheight = r[1]

home = [cropframewidth, int((cropframeheight+smallframeheight)/2)]




while True:
    _, frame = cap.read()
    
    frame = cv2.resize(frame, (0,0), fx = factor, fy=factor)

    hsv = cv2.cvtColor(frame,  cv2.COLOR_BGR2HSV)
    
   
    maskp = cv2.inRange(hsv, lower_pink, upper_pink)
    maskg = cv2.inRange(hsv, lower_green, upper_green)
   
    blurp = cv2.GaussianBlur(maskp, (15,15),0)
    blurg = cv2.GaussianBlur(maskg, (15,15),0)
    
    kernelOpen=np.ones((1,1))
    kernelClose=np.ones((20,20))
    

    maskOpenp=cv2.morphologyEx(blurp,cv2.MORPH_OPEN,kernelOpen)
    maskClosep=cv2.morphologyEx(maskOpenp,cv2.MORPH_CLOSE,kernelClose)
    
    maskOpeng=cv2.morphologyEx(blurg,cv2.MORPH_OPEN,kernelOpen)
    maskCloseg=cv2.morphologyEx(maskOpeng,cv2.MORPH_CLOSE,kernelClose)
    
    
    _, disc = cv2.threshold(maskOpenp, 3, 255, cv2.THRESH_BINARY)
    
    h, stick = cv2.threshold(maskOpeng, 3, 255, cv2.THRESH_BINARY)
    #cv2.imshow("maskOpen",maskOpen)
    
    
#    maskFinal=maskClose
#    conts,h=cv2.findContours(maskFinal.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
#    
#    for i in range(len(conts)):
#        x,y,w,h=cv2.boundingRect(conts[i])
#        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255), 2)
  #-------------------------------------------------------------------  
    xsstick = np.where(stick != 0)[1]
    ysstick = np.where(stick != 0)[0]
    
    if np.size(xsstick) >= 1:
    
        xstdstick = np.std(xsstick)
        ystdstick = np.std(ysstick)

        x_init_avgstick = np.mean(xsstick)
        y_init_avgstick = np.mean(ysstick)
    
    # possibly if std is too high, we can consider that we have a problem?

        xsstick = [x for x in xsstick if x <= x_init_avgstick+xstdstick or x >= x_init_avgstick-xstdstick]
        ysstick = [y for y in ysstick if y <= y_init_avgstick+xstdstick or y >= y_init_avgstick-xstdstick]
    
    
        
    
        xavgstick = int(round(np.mean(xsstick)))
        yavgstick = int(round(np.mean(ysstick)))
        
        frame = cv2.circle(frame, (xavgstick,yavgstick), 3, (0,255,0), 2)
    else:
         xavgstick = 0
         yavgstick = 0
#-----------------------------------------------------------------------------    
    xs = np.where(disc != 0)[1]
    ys = np.where(disc != 0)[0]
    
    if np.size(xs) >= 1:
    
        xstd = np.std(xs)
        ystd = np.std(ys)

        x_init_avg = np.mean(xs)
        y_init_avg = np.mean(ys)
    
    # possibly if std is too high, we can consider that we have a problem?

        xs = [x for x in xs if x <= x_init_avg+xstd or x >= x_init_avg-xstd]
        ys = [y for y in ys if y <= y_init_avg+xstd or y >= y_init_avg-xstd]
    
    
        
    
        xavg = int(round(np.mean(xs)))
        yavg = int(round(np.mean(ys)))
        
        trajx.append(xavg)
        trajy.append(yavg)
        xl = 0
        yl = 0
        
        if len(trajx) >= confidence:
            

            
            if (np.abs(trajx[0] - trajx[-1]) > 1 ) or (np.abs(trajy[0] - trajy[-1]) > 1 ) and (90 > np.abs(trajy[0] - trajy[-1])) and (90 > np.abs(trajx[0] - trajx[-1])):
                
                if (0 <= xavg <= framewidth) and (0 <= yavg <= frameheight):
                    xl, yl = trajectory(trajx, trajy)
            
#                    if np.size(xl) > 1:
#                        frame = cv2.line(frame, (trajx[len(trajx)-1], trajy[len(trajx)-1]), (xl[0], yl[0]), (0,0,255), 2)
#                        frame = cv2.line(frame, (xl[0], yl[0]), (xl[1], yl[1]), (0,0,255), 2)
#                
#                    
#                    else:
                    frame = cv2.line(frame, (trajx[len(trajx)-1], trajy[len(trajx)-1]), (xl, yl), (0,0,255), 2)
                    
                    if xavgstick and xl == home[0] and cropframeheight >= yl >= smallframeheight:
                        control(xl, yl, xavgstick, yavgstick, xavg)
                        #print('control')
                    else:
                        if xavgstick:
                            hm(xavgstick , yavgstick)
                            #print('home')
                        
                else:
                    xl = 0
                    y1 = 0
            trajx.pop(0)
            trajy.pop(0)
            
    
        frame = cv2.circle(frame, (xavg,yavg), 3, (0,0,255), 2)
    

    
    if xavgstick and (xl != home[0] or cropframeheight < yl or yl < smallframeheight):
        hm(xavgstick , yavgstick)
        #print('home')
        
    if not xavgstick and not xs:
        off()
        #print('off')
    if not xavgstick and xs:
        off()
        #print('off')
        
    if xavgstick and not xs:
        hm(xavgstick , yavgstick)
        #print('home')
    
    frame = cv2.rectangle(frame, (r[0], r[1]),(r[0] + r[2], r[1] + r[3]), (250,  30, 0), 2)
    frame = cv2.circle(frame, (home[0],home[1]), 3, (0,0,0), 2)
    
    #frame = cv2.resize(frame, (0,0), fx = 1/factor, fy=1/factor)
    if record:
        out.write(frame)
    cv2.imshow('frame', frame)
    t.append(tm.time())
    #cv2.imshow('mask', mask)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

cap.release()
cv2.destroyAllWindows()
GPIO.cleanup()

if record:
    out.release()

t = np.asarray(t)
print(np.average(1/np.diff(t)))
