import numpy as np
from body_part_angle import BodyPartAngle
from utils import *


class TypeOfExercise(BodyPartAngle):
    def __init__(self, landmarks):
        super().__init__(landmarks)

    def push_up(self, counter, status):
        left_arm_angle = self.angle_of_the_left_arm()  # 获取左臂的角度
        right_arm_angle = self.angle_of_the_left_arm()  # 获取右臂的角度
        avg_arm_angle = (left_arm_angle + right_arm_angle) // 2  # 计算两臂角度的平均值

        if status:  # 如果当前状态为True，表示用户正在向下做俯卧撑
            if avg_arm_angle < 70:  # 如果平均臂角度小于70度，表示俯卧撑已到达最低点
                counter += 1  # 计数器加1，记录一个俯卧撑的完成
                status = False  # 状态变为False，表示用户需要返回到起始位置
        else:  # 如果当前状态为False，表示用户正在向上恢复到起始位置
            if avg_arm_angle > 160:  # 如果平均臂角度大于160度，表示用户已经恢复到起始位置
                status = True  # 状态变为True，表示可以开始下一个俯卧撑

        return [counter, status]  # 返回当前的计数器值和状态

    def calculate_exercise(self, exercise_type, counter, status):
        if exercise_type == "push-up":
            counter, status = TypeOfExercise(self.landmarks).push_up(
                counter, status)

        return [counter, status]
