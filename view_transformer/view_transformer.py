import numpy as np 
import cv2

class ViewTransformer():
    def __init__(self):
        #football ground dimensions are 105 x 68
        court_width = 68 
        court_length = 23.32 #5.83 *4 [52.5/9 =5.83] divinding by 9 because one side is divided into 9 equal recetangles

        #defining the vertices of the trapezoid (refer the png image)
        self.pixel_vertices = np.array([[110, 1035], #these vertices can be estimated through AI tools
                               [265, 275], 
                               [910, 260], 
                               [1640, 915]])
        
        
        self.target_vertices = np.array([
            [0,court_width],
            [0, 0],
            [court_length, 0],
            [court_length, court_width]
        ])

        #casting it to be a float
        self.pixel_vertices = self.pixel_vertices.astype(np.float32)
        self.target_vertices = self.target_vertices.astype(np.float32)

        #initializing the perspective transformer
        self.perspective_transformer = cv2.getPerspectiveTransform(self.pixel_vertices, self.target_vertices)

    
    def transform_point(self,point):
        p = (int(point[0]),int(point[1]))
        is_inside = cv2.pointPolygonTest(self.pixel_vertices,p,False) >= 0 
        if not is_inside:
            return None

        reshaped_point = point.reshape(-1,1,2).astype(np.float32)
        transform_point = cv2.perspectiveTransform(reshaped_point,self.perspective_transformer)
        
        return transform_point.reshape(-1,2)

    #transform the adjusted points to the points after the perspective transformation
    def add_transformed_position_to_tracks(self,tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info['position_adjusted']
                    position = np.array(position)
                    position_transformed = self.transform_point(position)
                    if position_transformed is not None:
                        position_transformed = position_transformed.squeeze().tolist()
                    tracks[object][frame_num][track_id]['position_transformed'] = position_transformed