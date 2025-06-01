from ultralytics import YOLO

model = YOLO('yolov8x') # x - extra large for best accuracy (more parameters)

results = model.predict(r'C:\Users\kingr\football_project\input_videos\08fd33_4.mp4',save = True)
# [save = true] basically writes annotated video to disk 

print(results[0]) #first frame
print('============================')
 
for box in results[0].boxes:
    print(box)
    
    
"""Why using of out-of-the-box object detection model ?
it is a quick way to bootstrap i.e you see what it "already knows"(pre trained classes like 0:person,
 1:sports,etc) basically identify its gaps like poor ball detection and false positives on the sideline so that
we can apply our own fine-tuning or data collection to improve its performance."""