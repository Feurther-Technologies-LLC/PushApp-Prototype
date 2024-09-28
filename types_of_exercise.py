import numpy as np
from body_part_angle import BodyPartAngle
from utils import *


class TypeOfExercise(BodyPartAngle):
    def __init__(self, landmarks):
        super().__init__(landmarks)

    def push_up(self, counter, status, down_threshold, up_threshold):
        # 确保 counter 是整数
        counter = int(counter) if isinstance(counter, str) else counter

        left_arm_angle = self.angle_of_the_left_arm()
        right_arm_angle = self.angle_of_the_right_arm()  # 修正：使用右臂角度函数
        avg_arm_angle = (left_arm_angle + right_arm_angle) / 2  # 使用浮点除法

        if status:  # 向下阶段
            if avg_arm_angle < down_threshold:
                counter += 1
                status = False  # 切换到向上阶段
        else:  # 向上阶段
            if avg_arm_angle > up_threshold:
                status = True  # 切换到向下阶段

        return counter, status

    def calculate_exercise(self, exercise_type, counter, status, down_threshold, up_threshold):
        # 处理 'N/A' 的情况
        if counter == 'N/A':
            counter = 0
        elif isinstance(counter, str):
            try:
                counter = int(counter)
            except ValueError:
                counter = 0

        if exercise_type == "push-up":
            return self.push_up(counter, status, down_threshold, up_threshold)
        # 在这里可以添加其他运动类型的处理
        return counter, status
