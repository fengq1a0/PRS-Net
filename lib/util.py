import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 四元数旋转
def rotate(para, point):
    tmp = np.array([0,point[0],point[1],point[2]])
    tmp = product(para, tmp)
    tmp = product(tmp, inverse(para))
    return tmp[1:4]


def product(q1, q2):
    return np.array([q1[0] * q2[0] - tf.reduce_sum(tf.multiply(q1[1:4], q2[1:4])),
                    q1[0] * q2[1] +q2[0] * q1[1] + q1[2] * q2[3] - q1[3] * q2[2],
                    q1[0] * q2[2] +q2[0] * q1[2] - q1[1] * q2[3] + q1[3] * q2[1],
                    q1[0] * q2[3] +q2[0] * q1[3] + q1[1] * q2[2] - q1[2] * q2[1]])

def inverse(q):
    qq = np.array([q[0],-1*q[1],-1*q[2],-1*q[3]])
    return qq / (tf.reduce_sum(tf.multiply(q,q)))

# 可视化工具
def visualize(para,points,path):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-4, 36)
    ax.set_ylim(-4, 36)
    ax.set_zlim(-4, 36)
    X = np.arange(-4, 36, 1)
    Y = np.arange(-4, 36, 1)
    X, Y = np.meshgrid(X, Y)
    if np.sum(para) > 0.01:
        a = float(para[0])
        b = float(para[1])
        c = float(para[2])
        d = float(para[3])
        Z = (-1 * d - a * X - b * Y) / c
        ax.plot_wireframe(X,Y,Z,rstride=5,cstride=5)
        ax.scatter(points[:,0],points[:,1],points[:,2],s=20)
    fig.savefig(path)
    plt.close()
