import cv2
import argparse
from utils import *
from config import Config
import mediapipe as mp
from body_part_angle import BodyPartAngle
from types_of_exercise import TypeOfExercise
from playsound import playsound
import threading  # 导入线程库

def play_sound(sound_file: str) -> None:
    threading.Thread(target=playsound, args=(
        sound_file,), daemon=True).start()


def setup_video_capture(video_source):
    cap = cv2.VideoCapture(video_source if video_source else 0)
    cap.set(3, Config.VIDEO_RESIZE_WIDTH)
    cap.set(4, Config.VIDEO_RESIZE_HEIGHT)
    return cap


def main(arg):
    # drawing body
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    # 设置音频文件路径
    sound_not_ready = "sound/NotReady.MP3"
    sound_push_up = "sound/PushUP!.MP3"
    sound_ready = "sound/Ready.MP3"

    # 播放音频的函数，使用线程避免阻塞
    cap = setup_video_capture(args["video_source"])
    cap.set(3, Config.VIDEO_RESIZE_WIDTH)  # width
    cap.set(4, Config.VIDEO_RESIZE_HEIGHT)  # height

    ready_status = False  # 用于标记 is_ready 的状态

    # setup mediapipe
    with mp_pose.Pose(min_detection_confidence=Config.DETECTION_CONFIDENCE,
                      min_tracking_confidence=Config.TRACKING_CONFIDENCE) as pose:

        counter = 0
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

            # init variables
            left_arm_angle, right_arm_angle = 0, 0
            plank_angle = 0
            is_prone = False
            is_plank = False

            # 判断是否识别到了人
            if results.pose_landmarks is None:
                # 无法检测姿势
                status = "N/A"
                left_arm_angle = 0
                right_arm_angle = 0
                continue

            landmarks = results.pose_landmarks.landmark

            # 判断人是否是站着的
            is_prone = ProneDetection(landmarks)
            if not is_prone:
                print("用户未进入俯卧式")
                continue

            # 判断是否进入了平板式
            angles = BodyPartAngle(landmarks)
            plank_angle = angles.angle_of_the_plank()
            is_plank = PlankDetection(plank_angle)
            if not is_plank:
                print("用户未进入平板式")
                continue

            # 进行俯卧撑检测
            left_arm_angle = angles.angle_of_the_left_arm()
            right_arm_angle = angles.angle_of_the_right_arm()

            new_counter, status = TypeOfExercise(landmarks).calculate_exercise(
                args["exercise_type"], counter, status, Config.PUSHUP_ARM_DOWN_THRESHOLD, Config.PUSHUP_ARM_UP_THRESHOLD)

            # 当counter增加时，播放PushUP!音频
            if new_counter > counter:
                play_sound(sound_push_up)
            counter = new_counter

            # 检查 is_ready 的状态变化
            is_ready = is_prone and is_plank
            # 当用户从未准备好（ready_status 是 False）变为准备好（is_ready 是 True）时,播放音频
            if is_ready and not ready_status:
                play_sound(sound_ready)
            elif not is_ready and ready_status:
                play_sound(sound_not_ready)
            ready_status = is_ready  # 更新 ready 状态

            # add text to image frame
            processed_frame = add_text_to_frame(frame,
                                                left_arm_angle,
                                                right_arm_angle,
                                                plank_angle,
                                                status,
                                                counter,
                                                is_ready)
            # Draw landmarks
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


if __name__ == "__main__":
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

    main(args)
