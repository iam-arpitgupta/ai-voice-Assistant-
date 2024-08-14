import cv2  

web_cam = cv2.Videocapture(1)

def web_cam_capture():
    if not web_cam.isOpened():
        print('Eroor : camers isnot opened / not found')
        exit()
        
        pzth = 'webcame.jpg
        ret, frame = web_cam.read()'
        cv2.imwrite(path,frame)
        
    web_cam_capture()
    
import pyttsx3

def speak(text):
    #engine 
    engine = pyttsx3.init()
    #set propery 
    engine.setProperty('rate' :'150')
    engine.setProperty('volumne',0.9)
    
    #say the text 
    engine.say(text)
    
    #wait for the speech to get finished 
    engine.runAndWait()
    
    