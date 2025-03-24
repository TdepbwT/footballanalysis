from ultralytics import YOLO
import supervision as sv
import pickle
import os
import numpy as np
import pandas as pd
import cv2
import sys

sys.path.append('../')
from footballanalysis.utils import get_centre_of_bbox, get_bbox_width, get_foot_position


class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def add_position_to_tracks(sekf, tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object == 'ball':
                        position = get_centre_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]['position'] = position

    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1, {}).get('bbox', []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        # Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: {"bbox": x}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i + batch_size], conf=0.1)
            detections += detections_batch
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)

        tracks = {
            "players": [],
            "referees": [],
            "ball": []
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}

            # Covert to supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Convert GoalKeeper to player object
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]

            # Track Objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}

                if cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}

            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox": bbox}

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_centre_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35 * width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width // 2
        x2_rect = x_center + rectangle_width // 2
        y1_rect = (y2 - rectangle_height // 2) + 15
        y2_rect = (y2 + rectangle_height // 2) + 15

        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect), int(y1_rect)),
                          (int(x2_rect), int(y2_rect)),
                          color,
                          cv2.FILLED)

            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10

            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text), int(y1_rect + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )

        return frame

    def draw_traingle(self, frame, bbox, color):
        y = int(bbox[1])
        x, _ = get_centre_of_bbox(bbox)

        triangle_points = np.array([
            [x, y],
            [x - 10, y - 20],
            [x + 10, y - 20],
        ])
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

        return frame

    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        # Draw a semi-transparent rectaggle 
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), -1)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        team_ball_control_till_frame = team_ball_control[:frame_num + 1]
        # Get the number of time each team had ball control
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 2].shape[0]
        team_1 = team_1_num_frames / (team_1_num_frames + team_2_num_frames)
        team_2 = team_2_num_frames / (team_1_num_frames + team_2_num_frames)

        cv2.putText(frame, f"Team 1 Ball Control: {team_1 * 100:.2f}%", (1400, 900), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 0), 3)
        cv2.putText(frame, f"Team 2 Ball Control: {team_2 * 100:.2f}%", (1400, 950), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 0), 3)

        return frame
    
    def draw_radars(self, frame, tracks, frame_num, radar_size=(400,250), position=(50,800)):
        """
        Draw radars for each player in the frame
        
        params:
        frame: np.array: frame to draw radars on
        tracks: dict: dictionary of tracks
        frame_num: int: frame number
        radar_size: tuple: size of radar
        position: tuple: position to draw radar on frame
        
        return:
        np.array: frame with radars drawn
        """
        
        # create blank image for radar
        radar_width, radar_height = radar_size
        radar = np.zeros((radar_height, radar_width, 3), dtype=np.uint8)
        
        # draw radar (green background, white lines)
        radar[:, :] = (0, 150, 0)  # Darker green for better visibility
        
        # draw outline
        margin = 10
        pitch_width = radar_width - 2 * margin
        pitch_height = radar_height - 2 * margin
        cv2.rectangle(radar, (margin, margin), (margin + pitch_width, margin + pitch_height), (255, 255, 255), 2)
        
        # draw halfway line
        cv2.line(radar, (margin + pitch_width // 2, margin), (margin + pitch_width // 2, margin + pitch_height), (255, 255, 255), 2)
        
        # draw center circle
        center_circle_radius = min(pitch_width, pitch_height) // 10
        cv2.circle(radar, (margin + pitch_width // 2, margin + pitch_height // 2), center_circle_radius, (255, 255, 255), 2)
        
        # draw penalty areas
        penalty_area_width = pitch_width // 5
        penalty_area_height = pitch_height // 3
        cv2.rectangle(radar, (margin, margin + pitch_height // 2 - penalty_area_height // 2), 
                     (margin + penalty_area_width, margin + pitch_height // 2 + penalty_area_height // 2), (255, 255, 255), 2)
        cv2.rectangle(radar, (margin + pitch_width - penalty_area_width, margin + pitch_height // 2 - penalty_area_height // 2), 
                     (margin + pitch_width, margin + pitch_height // 2 + penalty_area_height // 2), (255, 255, 255), 2)
        
        # map player positions to radar
        if frame_num < len(tracks["players"]):
            # get all players in current frame
            players = tracks["players"][frame_num]
            
            # map from pitch coordinates to radar coordinates
            for player_id, player in players.items():
                if "position_adjusted" in player:
                    player_position = player.get("position_adjusted", [0, 0])
                    
                    # Normalize position to radar coordinate space
                    # Assuming position_adjusted is in range [-1, 1] for both x and y
                    # Convert to [0, 1] range first
                    norm_x = (player_position[0] + 1) / 2
                    norm_y = (player_position[1] + 1) / 2
                    
                    # Scale to radar coords - from normalized [0,1] to actual radar coordinates
                    x = int(margin + (norm_x * pitch_width))
                    y = int(margin + (norm_y * pitch_height))
                    
                    # ensure player is within radar bounds
                    x = max(margin, min(x, margin + pitch_width))
                    y = max(margin, min(y, margin + pitch_height))
                    
                    # draw player
                    color = player.get("team_color", (0, 0, 255))
                    cv2.circle(radar, (x, y), 5, color, -1)
                    cv2.circle(radar, (x, y), 5, (255, 255, 255), 1)
                    
                    # draw player id
                    cv2.putText(radar, str(player_id), (x + 6, y + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # draw ball
            if 1 in tracks["ball"][frame_num] and "position_adjusted" in tracks["ball"][frame_num][1]:
                ball_position = tracks["ball"][frame_num][1].get("position_adjusted", [0, 0])
                
                # Normalize ball position
                norm_ball_x = (ball_position[0] + 1) / 2
                norm_ball_y = (ball_position[1] + 1) / 2
                
                ball_x = int(margin + (norm_ball_x * pitch_width))
                ball_y = int(margin + (norm_ball_y * pitch_height))
                
                # ensure ball is within radar bounds
                ball_x = max(margin, min(margin + pitch_width, ball_x))
                ball_y = max(margin, min(margin + pitch_height, ball_y))
                
                # draw ball
                cv2.circle(radar, (ball_x, ball_y), 3, (255, 255, 255), -1)
        
        # Create overlay for the main frame
        overlay = frame.copy()
        x, y = position
        
        # Draw background rectangle
        cv2.rectangle(overlay, (x, y), (x + radar_width, y + radar_height), (0, 0, 0), -1)
        
        # Place radar on overlay
        overlay[y:y + radar_height, x:x + radar_width] = radar
        
        # Apply transparency
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Add title text
        cv2.putText(frame, "Top-Down View", (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
        return frame

    def draw_annotations(self, video_frames, tracks, team_ball_control):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            # Draw Players
            for track_id, player in player_dict.items():
                color = player.get("team_color", (0, 0, 255))
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)

                if player.get('has_ball', False):
                    frame = self.draw_traingle(frame, player["bbox"], (0, 0, 255))

            # Draw Referee
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0, 255, 255))

            # Draw ball 
            for track_id, ball in ball_dict.items():
                frame = self.draw_traingle(frame, ball["bbox"], (0, 255, 0))

            # Draw Team Ball Control
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)
            
            # Draw radars
            radar_size = (400, 250)
            radar_x = (frame.shape[1] - radar_size[0]) // 50
            radar_y = (frame.shape[0] - radar_size[1]) - 30
            frame = self.draw_radars(frame, tracks, frame_num, radar_size, (radar_x, radar_y))

            output_video_frames.append(frame)

        return output_video_frames