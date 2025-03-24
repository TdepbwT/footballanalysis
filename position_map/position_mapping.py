import cv2
import numpy as np

def draw_position_map(frame, tracks, frame_num, field_size=(500,300)):
    """
    draw 2d position map of players and ball on the field
    
    args:
    frame: np.array, frame of the video
    tracks: dict, tracks of players and ball
    frame_num: int, frame number
    field_size: tuple, size of the field
    
    returns:
    np.array, frame with 2d position map
    
    """
    # create empty field
    field = np.ones((field_size[1], field_size[0], 3), dtype=np.uint8) * 255
    
    # draw field lines
    field = cv2.rectangle(field, (0, 0), (field_size[0], field_size[1]), (0, 0, 0), 2)
    field = cv2.line(field, (field_size[0] // 2, field_size[1]), (0, 0, 0), 2)
    
    # map positions to field
    for player_id, player in tracks["players"][frame_num].items():
        position = player.get("position_adjusted", None)
        
        if position:
            x, y = int(position[0] * / 1920 * field_size[0]), int(position[1] / 1080 * field_size[1])
            team_color = player.get("team_color", (0, 0, 255))
            cv2.circle(field, (x, y), 5, team_color, -1)
            
    # map ball position to field
    for ball_id, ball in tracks["ball"][frame_num].items():
        position = ball.get("position_adjusted", None)
        
        if position:
            x, y = int(position[0] / 1920 * field_size[0]), int(position[1] / 1080 * field_size[1])
            cv2.circle(field, (x, y), 5, (0, 255, 0), -1)
    
    # resize field to fit frame
    field_resized = cv2.resize(field, (frame.shape[1], field_size[1]))
    
    # overlay field on frame
    combined_frame = np.vstack((frame, field_resized))
    
    return combined_frame
        


