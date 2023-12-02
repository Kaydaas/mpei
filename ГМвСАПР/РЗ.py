from math import *
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Набор 12-и опорных точек
control_points = np.array([[0, 4],
                           [2, 6],
                           [4, 3],
                           [6, 7],
                           [8, 2],
                           [10, 5],
                           [12, 1],
                           [14, 6],
                           [16, 3],
                           [18, 7],
                           [20, 2],
                           [22, 5]])

# Добавление двух дополнительных точек в началае и в конце массива опорных точек, чтобы кривая проходила через первую и последнюю точки
control_points_extended = []
control_points_extended.append([control_points[0][0] - (control_points[1][0] - control_points[0][0]), control_points[0][1] - (control_points[1][1] - control_points[0][1])])
for point in control_points:
    control_points_extended.append(point)
control_points_extended.append([control_points[len(control_points) - 1][0] - (control_points[len(control_points) - 2][0] - control_points[len(control_points) - 1][0]), control_points[len(control_points) - 1][1] - (control_points[len(control_points) - 2][1] - control_points[len(control_points) - 1][1])])
control_points_extended = np.array(control_points_extended)

n = len(control_points_extended) - 1
STEP = 1e-3
NUMBER_OF_SEGMENTS = n - 2

B_spline = []

def N0_3(t):
    return (1 - t)**3 / 6

def N1_3(t):
    return (3 * t**3 - 6 * t**2 + 4) / 6

def N2_3(t):
    return (-3 * t**3 + 3 * t**2 + 3 * t + 1) / 6

def N3_3(t):
    return t**3 / 6

# Рассчет масштаба для графика
def SetScale(ax, x, y, z):
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0

    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

def r(t):
    r = np.array([0, 0])

    for i in range(NUMBER_OF_SEGMENTS):

        ri = np.array([0, 0])
        ri = ri + N0_3(t) * np.array(control_points_extended[i])
        ri = ri + N1_3(t) * np.array(control_points_extended[i + 1])
        ri = ri + N2_3(t) * np.array(control_points_extended[i + 2])
        ri = ri + N3_3(t) * np.array(control_points_extended[i + 3])

        r = r + ri

        B_spline[i].append([ri[0], ri[1], 0])

    return r

# Координаты полигонов кривой
control_points_X = control_points[:, 0]
control_points_Y = control_points[:, 1]
control_points_Z = np.zeros_like(control_points_X)

# Создание 3Д графика
mpl.rcParams['toolbar'] = 'None'
fig = plt.figure('B-spline', figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Построение опорных точек на графике
ax.scatter(control_points_X, control_points_Y, control_points_Z, color='red', s=50, alpha=1, label='Опорные точки')
# Построение полигонов кривой на графике
ax.plot(control_points_X, control_points_Y, control_points_Z, label='Полигоны кривой')

# Изометрическая проекция
ax.set_proj_type('ortho')
ax.view_init(elev=30, azim=45)
ax.set_box_aspect([1, 1, 1])
ax.disable_mouse_rotation()

SetScale(ax, control_points_X, control_points_Y, control_points_Z)

for _ in range(NUMBER_OF_SEGMENTS):
    B_spline.append([])

# Рассчет B-сплайна
i = 0
while i <= 1:
    r(i)
    i += STEP

# Создание массивов с координатами для B-сплайна
B_spline_X = []
B_spline_Y = []
B_spline_Z = []

for segment in B_spline:
    segment = np.array(segment)
    B_spline_X.extend(segment[:, 0])
    B_spline_Y.extend(segment[:, 1])
    B_spline_Z.extend(segment[:, 2])

B_spline_X = np.array(B_spline_X)
B_spline_Y = np.array(B_spline_Y)
B_spline_Z = np.array(B_spline_Z)

# Построение B-сплайна
ax.plot(B_spline_X, B_spline_Y, B_spline_Z, label='B-сплайн', linewidth=2.5)

# Создание поверхности вращение на 100 относительно оси X
surface_of_revolution = []
surface_of_revolution.append([B_spline_X, B_spline_Y, B_spline_Z])

# Рассчет сплайнов для поверхности вращения
for angle in range(0, 100):
    surface_of_revolution.append([B_spline_X, B_spline_Y * cos(radians(angle)), B_spline_Y * sin(radians(angle))])

# Построение поверхности вращения
for i in range(len(surface_of_revolution)-1):
    ax.plot_surface(np.array([surface_of_revolution[i][0], surface_of_revolution[i+1][0]]),
                    np.array([surface_of_revolution[i][1], surface_of_revolution[i+1][1]]),
                    np.array([surface_of_revolution[i][2], surface_of_revolution[i+1][2]]),
                    alpha=0.5, color='blue')

ax.legend()
plt.show()
