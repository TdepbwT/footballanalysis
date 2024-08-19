from utils import read_video, save_video
from trackers import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
import numpy as np


def main():
    # read video
    video_frames = read_video("input_videos/08fd33_4.mp4")

    # initialize tracker
    tracker = Tracker("models/best.pt")

    # get object tracks
    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path="stubs/track_stubs.pkl")
    #get obj positions
    tracker.add_position_to_tracks(tracks)

    #camera movement estimation
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                              read_from_stub=True,
                                                                              stub_path="stubs/camera_movement_stub.pkl")
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)
    #interpolate ball positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # assign player teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0],tracks["players"][0])

    for frame_num, player_track in enumerate(tracks["players"]):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],
                                                 track["bbox"],
                                                 player_id)
            tracks["players"][frame_num][player_id]["team"] = team
            tracks["players"][frame_num][player_id]["team_color"] = team_assigner.team_colors[team]

    # assign ball possession
    player_ball_assigner = PlayerBallAssigner()
    team_ball_control = []

    for frame_num, player_track in enumerate(tracks["players"]):
        ball_bbox = tracks["ball"][frame_num][1]["bbox"]
        assigned_player = player_ball_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            if "ball_possession" not in tracks["players"][frame_num][assigned_player]:
                tracks["players"][frame_num][assigned_player]["ball_possession"] = False
            tracks["players"][frame_num][assigned_player]["ball_possession"] = True

            team = tracks["players"][frame_num][assigned_player].get("team")
            team_ball_control.append(team)
        else:
            # handle initial none values
            if team_ball_control:
                team_ball_control.append(team_ball_control[-1])
            else:
                # replace None with placeholder value
                team_ball_control.append(-1)

    team_ball_control = np.array(team_ball_control)

    # remove initial frames with placeholder to exclude from calculation
    team_ball_control = team_ball_control[team_ball_control != -1]


    # draw output
    # draw object tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)

    # draw camera movement
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)

    # save video
    save_video(output_video_frames, "output_videos/output_video.avi")


if __name__ == "__main__":
    main()
