from utils import read_video, save_video
from trackers import Tracker
import cv2
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
import numpy as np
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator

def main():
    #Read Video
    video_frames = read_video(r'C:\Users\kingr\football_project\input_videos\08fd33_4.mp4') #input video location on disk
    
    
    #Initialize Tracker
    tracker = Tracker(r'C:\Users\kingr\football_project\input_videos\models\best.pt') #best model extracted from model_inference using YOLOv8
    
    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                        stub_path=r'C:\Users\kingr\football_project\input_videos\stubs\tracks.pkl')
    
    #get object positions
    tracker.add_position_to_tracks(tracks)
    
    
    # camera movement estimator 
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                    read_from_stub=True,
                                                    stub_path=r"C:\Users\kingr\football_project\input_videos\stubs\camera_movement_stub.pkl"
    )
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks,camera_movement_per_frame)
    
    # View Transformer
    
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)
    
    # Interpolate ball Positions
    
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])
    
    
    # Speed and distance estimator
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)
    
    # Assign Player Teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0]) #frame number 0
    
    for frame_num , player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items(): #looping over each player in the frame
            team = team_assigner.get_player_team(video_frames[frame_num],
                                                 track['bbox'],
                                                 player_id)
            tracks['players'][frame_num][player_id]['team'] = team #adding team id to the track
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team] #adding team color to the track
    
    
    #save cropped image of a player
    for track_id, player in tracks['players'][0].items(): #frame number 0
        bbox = player['bbox']
        frame = video_frames[0]
        
        
        #crop bbox from the frame
        cropped_image =frame[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])] #from y1 to y2 and x1 to x2
        
        ''' bbox[0] = x1
        bbox[1] = y1
        bbox[2] = x2
        bbox[3] = y2'''

        #save the cropped image
        cv2.imwrite(r'C:\Users\kingr\football_project\output_videos\cropped_image.jpg', cropped_image)
        break
        
        
    #Assign Ball Acquisition
    
    player_assigner = PlayerBallAssigner()
    team_ball_control = [] # to show the ball possession by each team
    
    for frame_num , player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track,ball_bbox)
    
        if assigned_player != -1:
           tracks['players'][frame_num][assigned_player]['has_ball'] = True 
           team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
           
        else:
           team_ball_control.append(team_ball_control[-1]) 
    team_ball_control = np.array(team_ball_control)        
    
    # Draw Output
    # draw object tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks,team_ball_control)
    
    # Draw Camera movement
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames,camera_movement_per_frame)
    
    
    # Draw Speed and Distance
    speed_and_distance_estimator.draw_speed_and_distance(output_video_frames,tracks)

    
    #Save Video
    save_video(output_video_frames, r'C:\Users\kingr\football_project\output_videos\output.mp4')
    
if __name__ == '__main__':
    main()    
    
    
    
"""Summary:

read_video():	Reads all frames from a video file
save_video():	Saves a list of frames as a video """
