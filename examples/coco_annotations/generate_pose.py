import random
import matplotlib.pyplot as plt
import numpy as np
# import seaborn as sns
from scipy.spatial.transform import Rotation as R


import random
import math

num_points= 2



def get_random_translation(
    r_range: (float, float)=(1, 10),
    theta_range: (float, float)=(-math.pi/3, math.pi/3),
    azi_range: (float, float)=(-math.pi/3, math.pi/3)
) -> [float, float, float]:
    r = random.random() * (r_range[1] - r_range[0]) + r_range[0]
    theta = random.random() * (theta_range[1] - theta_range[0]) + theta_range[0]
    azi = random.random() * (azi_range[1] - azi_range[0]) + azi_range[0]

    x = r * math.sin(theta) * math.cos(azi)
    y = r * math.sin(theta) * math.sin(azi)
    z = r * math.cos(theta)
    return [x, y, z]

def get_random_rotation(
    roll_range: (float, float)=(-math.pi, math.pi),
    pitch_range: (float, float)=(-math.pi, math.pi),
    yaw_range: (float, float)=(-math.pi, math.pi)
) -> [float, float, float]:
    roll = random.random() * (roll_range[1] - roll_range[0]) + roll_range[0]
    pitch = random.random() * (pitch_range[1] - pitch_range[0]) + pitch_range[0]
    yaw = random.random() * (yaw_range[1] - yaw_range[0]) + yaw_range[0]
    return [roll, pitch, yaw]

def get_random_pose(
    r_range: (float, float)=(1, 10),
    theta_range: (float, float)=(-math.pi/3, math.pi/3),
    azi_range: (float, float)=(-math.pi/3, math.pi/3),
    roll_range: (float, float)=(-math.pi, math.pi),
    pitch_range: (float, float)=(-math.pi, math.pi),
    yaw_range: (float, float)=(-math.pi, math.pi)
):
    x, y, z = get_random_translation(r_range=r_range, theta_range=theta_range, azi_range=azi_range)
    roll, pitch, yaw = get_random_rotation(roll_range=roll_range, pitch_range=pitch_range, yaw_range=yaw_range)
    return [yaw*180/math.pi, pitch*180/math.pi, roll*180/math.pi, x, y, z]

robot_translation = []
robot_rotation = []

for i in range(num_points):
    robot_rotation.append(get_random_rotation(roll_range=(-0.01,0.01), yaw_range=(-np.pi, np.pi), pitch_range=(-0.01, 0.01)))
    robot_translation.append(get_random_translation(r_range=(0.5,15), theta_range=(np.pi/2, np.pi/2), azi_range=(np.pi/3, 2*np.pi/3)))

camera_pose = (0, 0, 0.576, 1.5707963267948966, 0, 0)
loc_rot = [a+b for a,b in zip (robot_translation, robot_rotation)]

obj_loc = ""
for item in loc_rot:
    obj_loc += f"{str(item).strip('[]').replace(',', '')}\n"

with open('./nihonbashi_positions', 'w') as f:
    f.write(obj_loc)


camera_loc = ""
for i in range(num_points):
    camera_loc += f"{str(camera_pose).strip('()').replace(',', '')}\n"

with open('./camera_positions', 'w') as f:
    f.write(camera_loc)
