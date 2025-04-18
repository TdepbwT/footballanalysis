from ultralytics import YOLO
import supervision as sv
import pickle
import os
import numpy as np
import pandas as pd
import cv2
import sys
from inference import get_model
from dotenv import load_dotenv
from sports.common.view import ViewTransformer
from sports.annotators.soccer import draw_pitch

sys.path.append('../')
from footballanalysis.utils import get_centre_of_bbox, get_bbox_width, get_foot_position

load_dotenv()
api_key = os.getenv("ROBOFLOW_API_KEY")

class Tracker:
    def __init__(self, model_path, field_model_id, api_key):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
        self.field_model = get_model(field_model_id, api_key)
     
        

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

    # detect pitch keypoints
    def detect_pitch_keypoints(self, frame, confidence_threshold=0.3):
        result = self.field_model.infer(frame, confidence = confidence_threshold)[0] # get the first result
        keypoints = sv.KeyPoints.from_inference(result) # convert to supervision KeyPoints format
        return keypoints
    
    # filter keypoints based on confidencec threshold#
    def filter_keypoints(self, keypoints, confidence_threshold=0.5):
        # create a mask of keypoints with confidence above threshold
        filter_mask = keypoints.confidence[0] > confidence_threshold
        # filtered points
        filtered_points = keypoints.xy[0][filter_mask]
        return keypoints
    
    # project players to pitch using keypoints
    def project_players_to_pitch(self, frame, tracks, keypoints):
        
        from sports.configs.soccer import SoccerPitchConfiguration
        import numpy as np
        pitch_config = SoccerPitchConfiguration()
        
        pitch_reference_points = pitch_config.edges
        
        if isinstance(pitch_reference_points, list):
            pitch_reference_points = np.array(pitch_reference_points)
            
        source_points = keypoints.xy[0]
        if isinstance(source_points, list):
            source_points = np.array(source_points)
            
        # check if source points and pitch reference points are of same length
        if len(source_points) != len(pitch_reference_points):
            min_points = min(len(source_points), len(pitch_reference_points))
            source_points = source_points[:min_points]
            pitch_reference_points = pitch_reference_points[:min_points]
            
        self.view_transformer = ViewTransformer(source=source_points,target=pitch_reference_points)
        # fit the view transformer to the keypoints and pitch reference points
        
        for frame_num, player_track in enumerate(tracks["players"]):
            for player_id, player_data in tracks["players"][frame_num].items():
                bbox = player_data["bbox"]
                position = get_foot_position(bbox)
                position_array = np.array([position])
                projected_position = self.view_transformer.transform_points(position_array)
                if hasattr(projected_position, 'shape') and projected_position.shape[0] > 0:
                    projected_position = projected_position[0]

                tracks["players"][frame_num][player_id]["position_adjusted"] = projected_position

    def project_objects_to_pitch(self, frame, tracks, frame_num):
        """
        Project players, referees, and ball onto the pitch using homography.
        """
        import numpy as np
        from sports.configs.soccer import SoccerPitchConfiguration
        from sports.common.view import ViewTransformer

        # Load pitch configuration
        pitch_config = SoccerPitchConfiguration()

        # Detect and filter keypoints
        keypoints = self.detect_pitch_keypoints(frame)
        filter_mask = keypoints.confidence[0] > 0.5
        frame_reference_points = keypoints.xy[0][filter_mask]
        pitch_reference_points = np.array(pitch_config.vertices)[filter_mask]

        # Ensure sufficient keypoints for homography
        if len(frame_reference_points) < 4 or len(pitch_reference_points) < 4:
            return None, None, None  # Skip projection if not enough keypoints

        # Perform homography transformation
        transformer = ViewTransformer(source=frame_reference_points, target=pitch_reference_points)

        # Project players, referees, and ball
        players_xy = []
        referees_xy = []
        ball_xy = []
        if frame_num < len(tracks["players"]):
            players_xy = [
                transformer.transform_points(np.array([player["position_adjusted"]]))
                for player in tracks["players"][frame_num].values()
                if "position_adjusted" in player
            ]
        if frame_num < len(tracks["referees"]):
            referees_xy = [
                transformer.transform_points(np.array([referee["position_adjusted"]]))
                for referee in tracks["referees"][frame_num].values()
                if "position_adjusted" in referee
            ]
        if frame_num < len(tracks["ball"]):
            ball_xy = [
                transformer.transform_points(np.array([ball["position_adjusted"]]))
                for ball in tracks["ball"][frame_num].values()
                if "position_adjusted" in ball
            ]

        return players_xy, referees_xy, ball_xy

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
    
    def draw_radars(self, frame, tracks, frame_num, radar_size=(400, 250), position=(50, 50)):
        """
        Draw radar with player positions and ball projected onto the pitch using the draw_pitch function.
        """
        import numpy as np
        import cv2
        from sports.annotators.soccer import draw_pitch, draw_points_on_pitch
        from sports.configs.soccer import SoccerPitchConfiguration
        from sports.common.view import ViewTransformer
        import supervision as sv

        # Load pitch configuration
        pitch_config = SoccerPitchConfiguration()

        # Create radar view (pitch visualization)
        radar = draw_pitch(
            config=pitch_config,
            background_color=sv.Color.from_hex("#FFFFFF"),  # White
            line_color=sv.Color.from_hex("#000000")  # Black
        )

        # Filter keypoints for accurate pitch projection
        keypoints = self.detect_pitch_keypoints(frame)
        filter_mask = keypoints.confidence[0] > 0.5
        frame_reference_points = keypoints.xy[0][filter_mask]
        pitch_reference_points = np.array(pitch_config.vertices)[filter_mask]

        # Ensure sufficient keypoints for homography
        if len(frame_reference_points) < 4 or len(pitch_reference_points) < 4:
            return frame  # Skip radar if not enough keypoints

        # Perform homography transformation
        transformer = ViewTransformer(source=frame_reference_points, target=pitch_reference_points)

        # Team color mapping (map team ID to sv.Color)
        # We'll store these to use consistently for all players
        team_color_map = {}
        
        # First pass: collect team colors
        if frame_num < len(tracks["players"]):
            for player_id, player in tracks["players"][frame_num].items():
                if "team" in player and "team_color" in player:
                    team = player["team"]
                    team_color = player["team_color"]
                    
                    # Convert BGR team_color to sv.Color
                    if isinstance(team_color, tuple) and len(team_color) == 3:
                        # BGR to RGB for sv.Color
                        r, g, b = team_color[2], team_color[1], team_color[0]
                        color = sv.Color.from_rgb_tuple((r, g, b))
                        team_color_map[team] = color

        # Set default team colors if not found
        if 1 not in team_color_map:
            team_color_map[1] = sv.Color.from_hex("#FF0000")  # Red
        if 2 not in team_color_map:
            team_color_map[2] = sv.Color.from_hex("#0000FF")  # Blue
        
        # Project players and ball onto the pitch
        if frame_num < len(tracks["players"]):
            for player_id, player in tracks["players"][frame_num].items():
                if "position_adjusted" in player:
                    # Get team
                    team = player.get("team", 0)
                    
                    # Get color from team_color_map
                    if team in team_color_map:
                        color = team_color_map[team]
                    else:
                        color = sv.Color.from_hex("#0000FF")  # Default blue
                    
                    # Project position to pitch
                    player_pos = transformer.transform_points(np.array([player["position_adjusted"]]))
                    
                    # Draw player on radar with team color
                    if player_pos.shape[0] > 0:
                        draw_points_on_pitch(
                            config=pitch_config,
                            xy=player_pos,
                            face_color=color,
                            edge_color=sv.Color.from_hex("#FFFFFF"),  # White edge
                            radius=8,
                            pitch=radar
                        )
        
        # Draw referees with unique color (yellow)
        if "referees" in tracks and frame_num < len(tracks["referees"]):
            for referee_id, referee in tracks["referees"][frame_num].items():
                if "position_adjusted" in referee:
                    referee_pos = transformer.transform_points(np.array([referee["position_adjusted"]]))
                    if referee_pos.shape[0] > 0:
                        draw_points_on_pitch(
                            config=pitch_config,
                            xy=referee_pos,
                            face_color=sv.Color.from_hex("#FFFF00"),  # Yellow for referees
                            edge_color=sv.Color.from_hex("#000000"),  # Black edge
                            radius=8,
                            pitch=radar
                        )
        
        # Draw ball
        if frame_num < len(tracks["ball"]):
            for ball_id, ball in tracks["ball"][frame_num].items():
                if "position_adjusted" in ball:
                    ball_pos = transformer.transform_points(np.array([ball["position_adjusted"]]))
                    if ball_pos.shape[0] > 0:
                        draw_points_on_pitch(
                            config=pitch_config,
                            xy=ball_pos,
                            face_color=sv.Color.from_hex("#FFFFFF"),  # White for ball
                            edge_color=sv.Color.from_hex("#000000"),  # Black edge
                            radius=5,
                            pitch=radar
                        )

        # Overlay radar on the main frame
        x, y = position
        radar_height, radar_width, _ = radar.shape
        
        # Ensure position is within frame bounds
        max_x = frame.shape[1] - radar_width
        max_y = frame.shape[0] - radar_height
        x = min(max(0, x), max_x)
        y = min(max(0, y), max_y)
        
        # Add title and border to radar
        cv2.rectangle(radar, (0, 0), (radar_width-1, radar_height-1), (0, 0, 0), 2)
        cv2.putText(radar, "Tactical View", (10, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        overlay = frame.copy()
        overlay[y:y + radar_height, x:x + radar_width] = radar
        alpha = 0.8
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        return frame

    def draw_annotations(self, video_frames, tracks, team_ball_control):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            # Draw players, referees, and ball
            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            for track_id, player in player_dict.items():
                color = player.get("team_color", (0, 0, 255))
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)

            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0, 255, 255))

            for track_id, ball in ball_dict.items():
                frame = self.draw_traingle(frame, ball["bbox"], (0, 255, 0))

            # Draw Team Ball Control
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)

            # Draw radar view
            radar_size = (400, 250)
            radar_x = (frame.shape[1] - radar_size[0]) // 2
            radar_y = frame.shape[0] - radar_size[1] - 30
            frame = self.draw_radars(frame, tracks, frame_num, radar_size, (radar_x, radar_y))

            output_video_frames.append(frame)

        return output_video_frames