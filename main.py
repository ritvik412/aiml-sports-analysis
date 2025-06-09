from utils import read_video, save_video
from trackers import Tracker

def main():
    #Read Video
    video_frames = read_video(r'C:\Users\kingr\football_project\input_videos\08fd33_4.mp4') 
    
    #Initialize Tracker
    tracker = Tracker(r'C:\Users\kingr\football_project\input_videos\models\best.pt')
    
    tracks = tracker.get_object_tracks(video_frames)
    
    
    #Save Video
    save_video(video_frames, r'C:\Users\kingr\football_project\output_videos\output.mp4')
    
if __name__ == '__main__':
    main()    
    
"""✅ Summary:

read_video():	Reads all frames from a video file
save_video():	Saves a list of frames as a video """