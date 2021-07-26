import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2

index = 0
arr = []
while True:
    cap = cv2.VideoCapture(index)
    if not cap.read()[0]:
        break
    else:
        arr.append(index)
    cap.release()
    index += 1
print(arr)

