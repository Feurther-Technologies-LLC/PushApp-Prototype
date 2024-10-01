class Config:
    # 姿势检测阈值

    # 俯卧撑检测阈值
    PUSHUP_ARM_DOWN_THRESHOLD = 80
    PUSHUP_ARM_UP_THRESHOLD = 150
    PUSHUP_PLANK_ANGLE_MIN = 160
    PUSHUP_PLANK_ANGLE_MAX = 200

    # 平板支撑检测阈值
    PLANK_THRESHOLD = 150

    # 站立检测阈值
    HEIGHT_DIFF_THRESHOLD = 0.5

    # 视频处理参数
    VIDEO_RESIZE_WIDTH = 800
    VIDEO_RESIZE_HEIGHT = 480

    # Mediapipe 检测阈值
    DETECTION_CONFIDENCE = 0.9
    TRACKING_CONFIDENCE = 0.9

    # 颜色 BGR
    COLOR_RED = (0, 0, 255)
    COLOR_GREEN = (0, 255, 0)
    COLOR_BLUE = (255, 0, 0)
