from ultralytics import YOLO
import supervision as sv
import pickle
import os
import numpy as np
import pandas as pd
import sys
import cv2
sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width, get_foot_position


class Tracker:
    def __init__(self,model_path): 
        self.model = YOLO(model_path)
        self.tracker =sv.ByteTrack()
        
    def add_position_to_tracks(self,tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items(): #we are getting the player position relative to the bounding box
                    bbox = track_info['bbox']
                    if object == 'ball':
                        position= get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]['position'] = position
    
    def interpolate_ball_positions(self,ball_positions): #estimating the ball positions in the frames where it is not detected
        ball_positions = [x.get(1,{}).get('bbox',[]) for x in ball_positions] #ball position format converted to pandas dataframe
        # get 1 which is the track id 1 if no track id then an empty dictionary
        # then we want the bbox if no bbox then we put an empty list, and this empty list will be interpolated by the pandas dataframe
        df_ball_positions = pd.DataFrame(ball_positions,columns=["x1","y1","x2","y2"])
        
        #Interpolate the missing values
        df_ball_positions= df_ball_positions.interpolate() 
        
        #If missing detection is first one,then it wont interpolate, so we replicate the nearest detection we find  
        df_ball_positions = df_ball_positions.bfill() 
        
        ball_positions = [{1: {"bbox":x}} for x in df_ball_positions.to_numpy().tolist()] # 1 is the track id and value is going to be a dictionary of bounding boxes i.e x
        
        return ball_positions 
               
    def detect_frames(self,frames):
        batch_size = 20
        detections = []
        for i in range(0,len(frames),batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size],conf=0.1)
            detections += detections_batch
            
        return detections
    
    def get_object_tracks(self,frames,read_from_stub=False,stub_path=None):
        
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks 
            
        detections = self.detect_frames(frames)
        
        tracks={
            "players":[], 
            "referees":[],
            "ball":[]
        }
        for frame_num, detection in enumerate(detections):
            cls_names = detection.names 
            cls_names_inv = {v:k for k,v in cls_names.items()}
            
            
            #Convert to supervision detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)
            
            #convert goalkeeper to player object
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]
             
            #track objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)
            
            tracks["players"].append({}) # appending this dictionary which will have key as track ids and value is going to be the bounding box
            tracks["referees"].append({}) 
            tracks["ball"].append({}) 
            
            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist() #extracting bounding box
                cls_id = frame_detection[3]
                track_id = frame_detection[4]
            # 0 i.e first one is detection , 1 is mask, 2 is confidence, 3 is class id and 4 is tracker id

            # add tracking for players and refereed not for ball as it is just one object so bounding box enough for it    
                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox":bbox} #frame_num is the list index
                
                if cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox":bbox}
                
                    
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                
                if cls_id == cls_names_inv['ball']: # although track id  isn't needed as there is only one ball in the game (use 1 for it)
                    tracks["ball"][frame_num][1] = {"bbox":bbox}
                    #can use track_id to keep the code robust in case there are multiple balls
                 
        if stub_path is not None:
           with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)
                
                
                
        return tracks 
    
    def draw_ellipse(self,frame,bbox,color,track_id=None): # giving track_id a default type which is None
        
        y2 = int(bbox[3]) # we want the ellipse to be drawn at the bottom of the bounding box which is y2
        
        x_center,_= get_center_of_bbox(bbox)
        width= get_bbox_width(bbox)
        
        cv2.ellipse(frame,center=(x_center,y2),
                    axes=(int(width),int(0.35*width)), # minor and major axes of the ellips
                    angle=0.0,
                    startAngle=-45,endAngle=235, # this means that a little bit of the ellipse isn't drawn and that will be visible in the output
                    color=color,
                    thickness=2,
                    lineType=cv2.LINE_4
                    
        )
        
        rectangle_width = 40
        rectangle_height = 20
        
        x1_rect = x_center - rectangle_width // 2
        x2_rect = x_center + rectangle_width // 2
        y1_rect = (y2 - rectangle_height//2) + 15 # 15 which is just a random buffer to make things neater
        y2_rect = (y2 + rectangle_height//2) + 15
        
        if track_id is not None:
            cv2.rectangle(frame, 
                          (int(x1_rect), int(y1_rect)), 
                          (int(x2_rect), int(y2_rect)), 
                          color,
                          cv2.FILLED)

            x1_text = x1_rect + 12 # a little bit padding which is 12 pixels
            if track_id >99:
                x1_text -=10
                
            cv2.putText(frame,
                        f"{track_id}",
                        (int(x1_text), int(y1_rect + 15)), # y1_rect + 15 is just a little bit padding to make it look better
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, #font size
                        (0,0,0), #black color
                        2, #thickness
                    )
            
        
        return frame
    
    
    def draw_triangle(self,frame,bbox,color): # triangle's three points are (x,y),(x-10,y-20),(x+10,y-20)
        y = int(bbox[1]) #need traingle to be on top of the ball
        x,_ = get_center_of_bbox(bbox) 
        
        triangle_points = np.array([
                [x,y],
                [x-10,y-20],
                [x+10,y-20]
        ])
        cv2.drawContours(frame,[triangle_points],0,color,cv2.FILLED)
        cv2.drawContours(frame,[triangle_points],0,(0,0,0),2) #drawing the outline of the triangle in black color
        
        return frame
    
    def draw_team_ball_control(self,frame,frame_num,team_ball_control):
        # Draw a semi transparent rectangle to show team ball control
        overlay = frame.copy() #helps with the transparency
        cv2.rectangle(overlay,(1350,850),(1900,970),(255,255,255),-1) # -1 for filled or we can write cv2.FILLED
        alpha = 0.4 # for the transparency i.e 40%
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame) # added a semi transparent rectangle
        
        #Calculate the % of time that a team has the ball
        
        team_ball_control_till_frame =  team_ball_control[:frame_num+1]
        
        # Get the number of time each team had ball control
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==2].shape[0]
        team_1 = team_1_num_frames/(team_1_num_frames+team_2_num_frames)
        team_2 = team_2_num_frames/(team_1_num_frames+team_2_num_frames)
        
        cv2.putText(frame, f"Team 1 Ball Possession: {team_1*100:.2f}%",(1400,900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3) # percentage(limit it to 2 decimal points), position,font,filled, color, thickness
        cv2.putText(frame, f"Team 2 Ball Possession: {team_2*100:.2f}%",(1400,950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
        
        return frame
    
    
    def draw_annotations(self,video_frames,tracks,team_ball_control):
        output_video_frames = []
        for frame_num,frame in enumerate(video_frames):
            
            frame= frame.copy()
            
            player_dict =tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num] 
            referee_dict = tracks["referees"][frame_num]
            
            # Draw Players
            for track_id,player in player_dict.items():
                color = player.get("team_color", (0,0,255)) # default color is blue if team color is not specified
                frame = self.draw_ellipse(frame, player["bbox"],color,track_id) #BGR colors can be given using lists or numpy arrays. Here, (0,0,255) is the red color
            
                if player.get('has_ball',False):
                    frame = self.draw_triangle(frame, player["bbox"],(0,0,255))
                    
            #Draw Referee
            for _, referee in referee_dict.items(): #not using track_id here as we are not interested in drawing the track id for referees 
                frame = self.draw_ellipse(frame, referee["bbox"],(0,255,255))
                
            #Draw Ball
            for tracker_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["bbox"], (0,255,0))
                # Here, (0,255,0) is the green color
            
            
            # Draw Team Ball Possession
            frame = self.draw_team_ball_control(frame,frame_num,team_ball_control)
                
            output_video_frames.append(frame)
            
        return output_video_frames
    
    