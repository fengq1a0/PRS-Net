import os
import numpy as np
from lib.mesh import Mesh
from lib.binvox_rw import read_as_3d_array

path = "data\\ShapeNetCore.v1\\02691156\\"
flist = sorted(os.listdir(path))

def sample(m):
    area = []
    ans = []
    for i in range(m.f.shape[0]):
        t1 = m.v[m.f[i,1]]-m.v[m.f[i,0]]
        t2 = m.v[m.f[i,2]]-m.v[m.f[i,0]]
        area.append(np.linalg.norm(np.cross(t1,t2)))
    area = np.array(area)
    for i in range(1,area.shape[0],1):
        area[i]+=area[i-1]
    area/=area[-1]
    area*=1000
    area[-1] = 1000
    ind = 0
    for i in range(1000):
        while (area[ind]<i): ind+=1
        t1 = 1-np.sqrt(np.random.rand())
        t2 = (1-t1)*np.random.rand()
        t3 = 1 - t1 - t2
        res = m.v[m.f[ind,0]]*t1+m.v[m.f[ind,1]]*t2+m.v[m.f[ind,2]]*t3
        ans.append(res)
    return np.array(ans)

def get_index(p):
    ans = np.zeros((32,32,32))
    num = 1/32
    i = 1/64
    for t1 in range(32):
        j = 1/64
        for t2 in range(32):
            k = 1/64
            for t3 in range(32):
                dis = np.linalg.norm(p-np.array([i,j,k]),axis=1,keepdims=True)
                ans[t1,t2,t3] = np.argmin(dis)
                i+=num
                j+=num
                k+=num
    return ans

points = []
voxels = []
nindex = []

cnt = 0
for f in flist:
    
    os.system("data\\binvox.exe -d 32 "+path+f+"\\model.obj && move "
            +path+f+"\\model.binvox data\\vox\\"+str(cnt).zfill(3)+".binvox")
    
    with open("data\\vox\\"+str(cnt).zfill(3)+".binvox","rb") as voxf:
        vox = read_as_3d_array(voxf)
        m = Mesh(path+f+"\\model.obj")
        m.v-=vox.translate
        m.v/=vox.scale
        tmp = sample(m)
        nindex.append(get_index(tmp))
        points.append(tmp)
        voxels.append(vox.data)
    print(cnt)

    cnt += 1

points = np.array(points)
voxels = np.array(voxels)
nindex = np.array(nindex)
print(points.shape)
print(voxels.shape)
print(nindex.shape)
np.save("data\\points.npy",points)
np.save("data\\voxels.npy",voxels)
np.save("data\\nindex.npy",nindex)
