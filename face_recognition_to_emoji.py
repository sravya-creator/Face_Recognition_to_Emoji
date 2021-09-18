import matplotlib.pyplot as plt
import numpy as np
import cv2
import pyautogui
from deepface import DeepFace
import emoji


#code to detect face
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

video = cv2.VideoCapture(0)
def rescale_frame(frame,percent=360):
    width=int(frame.shape[1]*percent/100)
    height=int(frame.shape[0]*percent/100)
    dim=(width,height)
    return cv2.resize(frame,dim,interpolation=cv2.INTER_AREA)

print("Press q to take screenshot")


while True:
    check, frame = video.read()
    faces = face_cascade.detectMultiScale(frame,
                                          scaleFactor=1.1, minNeighbors=5)
    for x,y,w,h in faces:
        frame = cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 3)
    frame360=rescale_frame(frame,percent=360)
    cv2.imshow('Press q to take screenshot', frame360)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break
#code to take screenshot
image = pyautogui.screenshot()
image = cv2.cvtColor(np.array(image),cv2.COLOR_RGB2BGR)
cv2.imwrite("image1.jpeg", image)
video.release()
cv2.destroyAllWindows()
 

'''#code to crop the face
img = cv2.imread('image1.jpeg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#backtorgb = cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)


for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    sub_face = img[y:y+h, x:x+w]
    face_file_name = "crop_image.jpg"
    plt.imsave(face_file_name, sub_face)
plt.imshow(sub_face)'''



img1=cv2.imread("image1.jpeg")
#img1=cv2.imread("C:\\Users\\hp\\Downloads\\sad.jpeg")
plt.imshow(img1[:,:,::-1])
plt.show()
result=DeepFace.analyze(img1,actions=['emotion'])
if(result.get('dominant_emotion')=='happy'):
   print(emoji.emojize(":grinning_face_with_big_eyes:"))
elif(result.get('dominant_emotion')=='sad'):
   print(emoji.emojize(":frowning_face:"))
elif(result.get('dominant_emotion')=='surprise'):
   print(emoji.emojize(":hushed_face:"))
elif(result.get('dominant_emotion')=='angry'):
   print(emoji.emojize(":angry_face:"))
elif(result.get('dominant_emotion')=='disgust'):
   print(emoji.emojize(":nauseated_face:"))
elif(result.get('dominant_emotion')=='fear'):
   print(emoji.emojize(":fearful_face:"))
else:
    print("Neutral")
#print(result)
#print(result.get('dominant_emotion'))
