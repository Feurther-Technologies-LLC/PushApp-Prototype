import mediapipe as mp
import pandas as pd
import numpy as np
import cv2
from config import Config

mp_pose = mp.solutions.pose


# returns an angle value as a result of the given points
def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) -\
        np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    # check cord sys area
    if angle > 180.0:
        angle = 360 - angle

    return angle


# return body part x,y value
def detection_body_part(landmarks, body_part_name):
    return [
        landmarks[mp_pose.PoseLandmark[body_part_name].value].x,
        landmarks[mp_pose.PoseLandmark[body_part_name].value].y,
        landmarks[mp_pose.PoseLandmark[body_part_name].value].visibility
    ]


def score_table(exercise, counter, status):
    score_table = cv2.imread("./images/score_table.png")

    # 处理None值
    exercise_display = exercise.replace(
        "-", " ") if exercise is not None else "N/A"
    counter_display = str(counter) if counter is not None else "N/A"
    status_display = str(status) if status is not None else "N/A"

    cv2.putText(score_table, "Activity : " + exercise_display,
                (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (182, 158, 128), 2,
                cv2.LINE_AA)
    cv2.putText(score_table, "Counter : " + counter_display, (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (182, 158, 128), 2, cv2.LINE_AA)
    cv2.putText(score_table, "Status : " + status_display, (10, 135),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (182, 158, 128), 2, cv2.LINE_AA)
    cv2.imshow("Score Table", score_table)

# Determine colors based on angles


def get_color(angle, down_threshold, up_threshold):
    if angle == "N/A":
        return Config.COLOR_BLUE
    if angle < down_threshold:
        return Config.COLOR_GREEN  # Green for down position
    elif angle > up_threshold:
        return Config.COLOR_RED  # Red for up position
    else:
        return Config.COLOR_BLUE  # Blue for intermediate positions


def add_text_to_frame(frame,
                      left_arm_angle,
                      right_arm_angle,
                      plank_angle,
                      status,
                      counter,
                      is_ready,
                      ):
    # Helper function to format angle display
    def format_angle(angle):
        return str(round(angle, 1)) if isinstance(angle, (int, float)) else str(angle)

    left_color = get_color(
        left_arm_angle, Config.PUSHUP_ARM_DOWN_THRESHOLD, Config.PUSHUP_ARM_UP_THRESHOLD)
    right_color = get_color(
        right_arm_angle, Config.PUSHUP_ARM_DOWN_THRESHOLD, Config.PUSHUP_ARM_UP_THRESHOLD)

    plank_color = Config.COLOR_GREEN if plank_angle >= Config.PLANK_THRESHOLD else Config.COLOR_RED

    # Display the angles on the frame with color feedback and bolder font
    cv2.putText(frame, f'Left Arm: {format_angle(left_arm_angle)}',
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, left_color, 2)
    cv2.putText(frame, f'Right Arm: {format_angle(right_arm_angle)}',
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, right_color, 2)
    cv2.putText(frame, f'Plank Angle: {format_angle(plank_angle)}',
                (400, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, plank_color, 2)

    # Display push-up status with bolder font
    if status == "N/A":
        status_text = "N/A"
        status_color = (0, 0, 0)  # White for N/A
    else:
        status_text = "Down" if status else "Up"
        status_color = (0, 255, 0) if status else (
            0, 0, 255)  # Green for Down, Red for Up

    cv2.putText(frame, f'Status: {status_text}',
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

    ready_color = plank_color = Config.COLOR_GREEN if is_ready else Config.COLOR_RED

    # Display counter with bolder font and yellow color
    cv2.putText(frame, f'Count: {counter}',
                (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, f'Is_Ready: {is_ready}',
                (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, ready_color, 2)

    return frame


def isProne(landmarks) -> bool:
    left_ankle_y = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y
    right_ankle_y = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y
    left_shoulder_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y
    right_shoulder_y = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y

    ankle_y = (left_ankle_y + right_ankle_y) / 2
    shoulder_y = (left_shoulder_y + right_shoulder_y) / 2

    height_diff = ankle_y-shoulder_y
    if height_diff > Config.HEIGHT_DIFF_THRESHOLD:  # threshold 是一个合理的数值，如0.5
        return False
    else:
        return True


def isPlank(plank_angle) -> bool:
    return plank_angle >= Config.PLANK_THRESHOLD
