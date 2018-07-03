from imageai.Detection import ObjectDetection
import os
from PIL import Image
import cv2

execution_path=os.getcwd()
#print(execution_path)

detector=ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath(os.path.join(execution_path,'resnet50_coco_best_v2.0.1.h5'))
#detector.setModelPath('resnet50_coco_best_v2.1.0.h5')
detector.loadModel()
#detections=detector.detectObjectsFromImage(input_image='image.jpg',output_image_path='imagenew.jpg')

detections=detector.detectObjectsFromImage(input_image=os.path.join(execution_path,'image.jpg'),
                                          output_image_path=os.path.join(execution_path,'imagenew.jpg'))
img=Image.open('imagenew.jpg')
img.show()

'''
#可以一帧一帧读取视频
#cap.read()读取返回来的一帧图像是矩阵，不能用Image.save()保存，可以用CV2.imwrite()保存
#先保存下来，以符合detectObjectFromimage()参数输入格式

cap = cv2.VideoCapture(0)
while (1):
    ret, img = cap.read()
    cv2.imwrite('image.jpg',img)
    detections = detector.detectObjectsFromImage(input_image='image.jpg',output_image_path=os.path.join(execution_path, 'imagenew.jpg'))
    frame= Image.open('imagenew.jpg')
    frame.show()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
'''
for eachObject in detections:
    print(eachObject['name']+':'+eachObject['percentage_probability'])

