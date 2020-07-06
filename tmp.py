import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
a = np.load("res.npy")
b = np.load("data\\points.npy")

def visualize(para,points,path):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-4, 36)
    ax.set_ylim(-4, 36)
    ax.set_zlim(-4, 36)
    ax.scatter(points[:,0],points[:,1],points[:,2],s=20,c='b')
    ax.scatter(para[:,0],para[:,1],para[:,2],s=20,c='r')
    fig.savefig(path)
    plt.close()


for i in range(a.shape[0]):
    tmp1 = a[i]
    tmp2 = b[i]
    visualize(tmp1[:,0,:]*32,tmp2*32,"result\\"+str(i).zfill(3)+"_0.png")
    visualize(tmp1[:,1,:]*32,tmp2*32,"result\\"+str(i).zfill(3)+"_1.png")
    visualize(tmp1[:,2,:]*32,tmp2*32,"result\\"+str(i).zfill(3)+"_2.png")
    