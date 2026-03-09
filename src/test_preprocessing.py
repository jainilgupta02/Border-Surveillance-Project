from preprocessing import extract_frames

video = 0

for frame_id, frame in extract_frames(video):
    print(frame_id, frame.shape)