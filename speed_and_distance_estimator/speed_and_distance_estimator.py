import cv2
import sys
import json
import os
import numpy as np

sys.path.append('../')
from footballanalysis.utils import measure_distance, get_foot_position


class SpeedAndDistance_Estimator():
    def __init__(self):
        self.frame_window = 5
        self.frame_rate = 24

    def add_speed_and_distance_to_tracks(self, tracks):
        total_distance = {}
        total_time = {}
        max_speed = {}
        average_speed = {}

        for object, object_tracks in tracks.items():
            if object == "ball" or object == "referees":
                continue

            for frame_num in range(0, len(object_tracks), self.frame_window):
                last_frame = min(frame_num + self.frame_window, len(object_tracks) - 1)
                
                for track_id in object_tracks[frame_num]:
                    if track_id not in object_tracks[last_frame]:
                        continue
                    
                    start_pos = object_tracks[frame_num][track_id].get('position_transformed')
                    end_pos = object_tracks[last_frame][track_id].get('position_transformed')
                    
                    if not start_pos or not end_pos:
                        continue
                    
                    distance = measure_distance(start_pos, end_pos)
                    time_elapsed = (last_frame - frame_num) / self.frame_rate
                    speed_mps = distance / time_elapsed
                    speed_kph = speed_mps * 3.6
                    team = object_tracks[frame_num][track_id].get('team', 'Unknown')
                    
                    total_distance.setdefault(object, {}).setdefault(track_id, 0.0)
                    total_time.setdefault(object, {}).setdefault(track_id, 0.0)
                    max_speed.setdefault(object, {}).setdefault(track_id, 0.0)
                    average_speed.setdefault(object, {}).setdefault(track_id, [])
                    
                    total_distance[object][track_id] += distance
                    total_time[object][track_id] += time_elapsed
                    max_speed[object][track_id] = max(max_speed[object][track_id], speed_kph)
                    average_speed[object][track_id].append(speed_kph)
                    
                    for f in range(frame_num, last_frame):
                        if track_id in tracks[object][f]:
                            tracks[object][f][track_id]['speed'] = speed_kph
                            tracks[object][f][track_id]['distance'] = total_distance[object][track_id]
                            tracks[object][f][track_id]['team'] = team
        
        for object, object_tracks in average_speed.items():
            for track_id, speeds in object_tracks.items():
                average_speed[object][track_id] = sum(speeds) / len(speeds) if speeds else 0.0
        
        self.save_statistics(total_distance, total_time, max_speed, average_speed)

    def save_statistics(self, total_distance, total_time, max_speed, average_speed):
        os.makedirs('output_videos', exist_ok=True)
        
        def convert_values(obj):
            if isinstance(obj, dict):
                return {str(k): convert_values(v) for k, v in obj.items()}  # Ensure keys are strings
            elif isinstance(obj, list):
                return [convert_values(i) for i in obj]
            elif isinstance(obj, (np.integer, np.floating)):  # Convert numpy numbers to Python types
                return obj.item()
            elif isinstance(obj, (int, float, str, bool, type(None))):
                return obj
            else:
                return str(obj)  # Convert unknown types to string

        player_data = {}
        for obj, obj_tracks in total_distance.items():
            for tid in obj_tracks:
                player_data[tid] = {
                    "total_distance": f"{total_distance[obj][tid]:.2f} m",
                    "total_time": f"{total_time[obj][tid]:.2f} s",
                    "max_speed": f"{max_speed[obj][tid]:.2f} km/h",
                    "average_speed": f"{average_speed[obj][tid]:.2f} km/h",
                }
        
        player_data = convert_values(player_data)  # Ensure all values are serializable

        with open('output_videos/player_statistics.json', 'w') as f:
            json.dump({"players": player_data}, f, indent=4)

    def draw_speed_and_distance(self, frames, tracks):
        output_frames = []
        for frame_num, frame in enumerate(frames):
            for object, object_tracks in tracks.items():
                if object in ["ball", "referees"]:
                    continue
                for track_id, track_info in object_tracks[frame_num].items():
                    speed = track_info.get('speed')
                    distance = track_info.get('distance')
                    if speed is None or distance is None:
                        continue
                    
                    bbox = track_info['bbox']
                    position = get_foot_position(bbox)
                    position = (int(position[0]), int(position[1] + 40))
                    
                    cv2.putText(frame, f"{speed:.2f} km/h", position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                    cv2.putText(frame, f"{distance:.2f} m", (position[0], position[1] + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            output_frames.append(frame)
        return output_frames
