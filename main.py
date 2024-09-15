# import packages
import cv2
import argparse
from utils import *
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

cap.set(3, 800)  # width
cap.set(4, 480)  # height

# setup mediapipe
with mp_pose.Pose(min_detection_confidence=0.9,
                  min_tracking_confidence=0.9) as pose:

    counter = 0  # movement of exercise
    status = True  # state of move
    while cap.isOpened():
        # Get video frame
        ret, frame = cap.read()
        frame = cv2.resize(frame, (800, 480), interpolation=cv2.INTER_AREA)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # do prediction
        frame_rgb.flags.writeable = False
        results = pose.process(frame_rgb)
        frame_rgb.flags.writeable = True

        if results.pose_landmarks is not None:
            landmarks = results.pose_landmarks.landmark

            # Initialize BodyPartAngle class
            angles = BodyPartAngle(landmarks)
            left_arm_angle = angles.angle_of_the_left_arm()
            right_arm_angle = angles.angle_of_the_right_arm()

            # Determine colors based on angles
            def get_color(angle):
                if angle < 70:
                    return (0, 255, 0)  # Green for down position
                elif angle > 160:
                    return (0, 0, 255)  # Red for up position
                else:
                    return (255, 0, 0)  # Blue for intermediate positions

            left_color = get_color(left_arm_angle)
            right_color = get_color(right_arm_angle)

            # Display the angles on the frame with color feedback and bolder font
            cv2.putText(frame, f'Left Arm: {int(left_arm_angle)}',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, left_color, 2)
            cv2.putText(frame, f'Right Arm: {int(right_arm_angle)}',
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, right_color, 2)

            # Process exercise counting
            counter, status = TypeOfExercise(landmarks).calculate_exercise(
                args["exercise_type"], counter, status)

            # Display push-up status with bolder font
            status_text = "Down" if status else "Up"
            status_color = (0, 255, 0) if status else (
                0, 0, 255)  # Green for Down, Red for Up
            cv2.putText(frame, f'Status: {status_text}',
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

            # Display counter with bolder font and yellow color
            cv2.putText(frame, f'Count: {counter}',
                        (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Render detections (for landmarks)
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(255, 255, 255),
                                   thickness=2,
                                   circle_radius=2),
            mp_drawing.DrawingSpec(color=(174, 139, 45),
                                   thickness=2,
                                   circle_radius=2),
        )

        cv2.imshow('Video', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            print("counter: " + str(counter))
            break

    cap.release()
    cv2.destroyAllWindows()
