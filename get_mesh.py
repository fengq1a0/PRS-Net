from lib.mesh import Mesh
import numpy as np
import os

path = "data\\ShapeNetCore.v1\\02691156\\"
flist = sorted(os.listdir(path))
cnt = 0
for f in flist:
    a = Mesh(path+f+"\\model.obj")
    r = np.random.rand(3)*2*np.pi
    R1 = np.array( [[1,0,0],
                    [0,   np.cos(r[0]),-1*np.sin(r[0])],
                    [0,   np.sin(r[0]),   np.cos(r[0])]])
    R2 = np.array( [[   np.cos(r[1]),0,np.sin(r[1])],
                    [0,1,0],
                    [-1*np.sin(r[1]),0,np.cos(r[1])]])
    R3 = np.array( [[np.cos(r[2]),-1*np.sin(r[2]),0],
                    [np.sin(r[2]),   np.cos(r[2]),0],
                    [0,0,1]])
    a.v = np.matmul(np.matmul(np.matmul(R1,R2),R3),a.v.T).T
    a.write_obj("data\\mesh\\"+str(cnt).zfill(3)+".obj")
    cnt+=1
