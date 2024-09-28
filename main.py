# import packages
import cv2
import argparse
from utils import *
from config import Config
import mediapipe as mp
from body_part_angle import BodyPartAngle
from types_of_exercise import TypeOfExercise

# setup argparse
ap = argparse.ArgumentParser()
ap.add_argument("-t",
                "--exercise_type",
                type=str,
                help='Type of activity to do',
                required=True)
ap.add_argument("-vs",
                "--video_source",
                type=str,
                help='Type of activity to do',
                required=False)
args = vars(ap.parse_args())

# drawing body
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# setting the video source
if args["video_source"] is not None:
    cap = cv2.VideoCapture(args["video_source"])
else:
    cap = cv2.VideoCapture(0)  # webcam

cap.set(3, Config.VIDEO_RESIZE_WIDTH)  # width
cap.set(4, Config.VIDEO_RESIZE_HEIGHT)  # height

# setup mediapipe
with mp_pose.Pose(min_detection_confidence=Config.DETECTION_CONFIDENCE,
                  min_tracking_confidence=Config.TRACKING_CONFIDENCE) as pose:

    counter = 0  # movement of exercise
    status = True  # state of move
    while cap.isOpened():
        # Get video frame
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(
                frame, (Config.VIDEO_RESIZE_WIDTH, Config.VIDEO_RESIZE_HEIGHT), interpolation=cv2.INTER_AREA)
        else:
            print("无法读取视频帧")
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # do prediction
        frame_rgb.flags.writeable = False
        results = pose.process(frame_rgb)
        frame_rgb.flags.writeable = True

        left_color = Config.COLOR_GREEN
        right_color = Config.COLOR_GREEN
        plank_color = Config.COLOR_GREEN
        left_arm_angle, right_arm_angle = 0, 0
        plank_angle = 0
        is_prone = False
        is_plank = False
        if results.pose_landmarks is not None:
            landmarks = results.pose_landmarks.landmark
            # 判断人是否是站着的
            is_prone = isProne(landmarks)
            if is_prone:
                # 判断是否进入了平板式
                is_plank = isPlank(landmarks)
                if is_plank:
                    # 进行俯卧撑检测
                    angles = BodyPartAngle(landmarks)
                    left_arm_angle = angles.angle_of_the_left_arm()
                    right_arm_angle = angles.angle_of_the_right_arm()
                    plank_angle = angles.angle_of_the_plank()

                    # Process exercise counting
                    counter, status = TypeOfExercise(landmarks).calculate_exercise(
                        args["exercise_type"], counter, status, Config.PUSHUP_ARM_DOWN_THRESHOLD, Config.PUSHUP_ARM_UP_THRESHOLD)

                else:
                    print("用户未进入平板式")
            else:
                print("用户未进入平板式")

        else:
            # 无法检测姿势
            status = "N/A"
            left_arm_angle = "N/A"
            right_arm_angle = "N/A"

        is_ready = is_prone and is_plank
        print(f"is_prone: {is_prone}   is_plank: {is_plank}")
        # add text to image frame
        processed_frame = add_text_to_frame(frame,
                                            left_arm_angle,
                                            right_arm_angle,
                                            plank_angle,
                                            status,
                                            counter,
                                            is_ready)
        # Render detections (for landmarks)
        mp_drawing.draw_landmarks(
            processed_frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(255, 255, 255),
                                   thickness=2,
                                   circle_radius=2),
            mp_drawing.DrawingSpec(color=(174, 139, 45),
                                   thickness=2,
                                   circle_radius=2),
        )

        cv2.imshow('Video', processed_frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            print("counter: " + str(counter))
            break

    cap.release()
    cv2.destroyAllWindows()
