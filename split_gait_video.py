import cv2
import numpy as np
import os

def split_gait_video(
        video_path,
        subject_id,
        condition="nm",
        angle="090",
        output_dir="splits",
        min_area=5000
    ):
    """
    Împarte un videoclip în treceri individuale și le salvează în format CASIA-B:
    <subject>_<condition>-<sequence>_<angle>.mp4
    Ex: 125_nm-01_090.mp4
    """

    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Eroare: nu pot deschide videoclipul.")
        return

    # FPS original
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"FPS original: {fps}")

    segment_idx = 1
    recording = False
    out = None

    # Background subtractor
    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50)

    # înainte de while
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        fgmask = fgbg.apply(frame)
        fgmask = cv2.medianBlur(fgmask, 5)

        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        moving = any(cv2.contourArea(cnt) > min_area for cnt in contours)

        # Start segment
        if moving and not recording:
            recording = True
            h, w = frame.shape[:2]
            frame_count = 0  # reset counter

            seq = f"{segment_idx:02d}"
            filename = f"{subject_id}_{condition}-{seq}_{angle}_gei.mp4"
            output_path = os.path.join(output_dir, filename)

            out = cv2.VideoWriter(
                output_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (w, h)
            )

            print(f"Started segment {segment_idx}: {filename}")
            segment_idx += 1

        # Write frames
        if recording:
            out.write(frame)
            frame_count += 1

        # Stop segment
        if not moving and recording:
            recording = False
            out.release()
            print("Stopped segment")

            duration = frame_count / fps
            if duration < 3.0:
                print(f"Segment prea scurt ({duration:.2f}s) șters.")
                os.remove(output_path)
            else:
                print(f"Segment pastrat ({duration:.2f}s).")

    cap.release()
    print("Done splitting video.")



# split_gait_video("D:\\Facultate\\Anul3\\Sem1\\PI-p\\GaHu-video\\Originals\\S001.avi", subject_id="125", condition="nm", angle="090")
# #split_gait_video("D:\\Facultate\\Anul3\\Sem1\\PI-p\\GaHu-video\\Originals\\S002.avi", subject_id="126", condition="nm", angle="090")
# split_gait_video("D:\\Facultate\\Anul3\\Sem1\\PI-p\\GaHu-video\\Originals\\S003.avi", subject_id="127", condition="nm", angle="090")
# split_gait_video("D:\\Facultate\\Anul3\\Sem1\\PI-p\\GaHu-video\\Originals\\S004.avi", subject_id="128", condition="nm", angle="090")
# split_gait_video("D:\\Facultate\\Anul3\\Sem1\\PI-p\\GaHu-video\\Originals\\S005.avi", subject_id="129", condition="nm", angle="090")
# split_gait_video("D:\\Facultate\\Anul3\\Sem1\\PI-p\\GaHu-video\\Originals\\S006.avi", subject_id="130", condition="nm", angle="090")
# split_gait_video("D:\\Facultate\\Anul3\\Sem1\\PI-p\\GaHu-video\\Originals\\S007.avi", subject_id="131", condition="nm", angle="090")
#split_gait_video("D:\\Facultate\\Anul3\\Sem1\\PI-p\\GaHu-video\\Originals\\S008.avi", subject_id="131", condition="nm", angle="090")
split_gait_video("D:\\Facultate\\Anul3\\Sem1\\PI-p\\GaHu-video\\Originals\\S009.avi", subject_id="131", condition="nm", angle="090")
for i in range(10, 17):
    split_gait_video(f"D:\\Facultate\\Anul3\\Sem1\\PI-p\\GaHu-video\\Originals\\S0{i}.avi", subject_id=f"{122+i}", condition="nm", angle="090")
