from ultralytics import YOLO

model = YOLO(r'C:\Users\kingr\football_project\input_videos\models\best.pt') 

results = model.predict(r'C:\Users\kingr\football_project\input_videos\08fd33_4.mp4',save = True)
# [save = true] basically writes annotated video to disk 

print(results[0]) #first frame
print('============================')
 
for box in results[0].boxes:
    print(box)