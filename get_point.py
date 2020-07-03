import os
import numpy as np
from lib.mesh import Mesh
from lib.binvox_rw import read_as_3d_array
import scipy.spatial

path = "data\\mesh\\"
flist = sorted(os.listdir(path))

def sample(m,nn):
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
    area*=nn
    area[-1] = nn
    ind = 0
    for i in range(nn):
        while (area[ind]<i): ind+=1
        t1 = 1-np.sqrt(np.random.rand())
        t2 = (1-t1)*np.random.rand()
        t3 = 1 - t1 - t2
        res = m.v[m.f[ind,0]]*t1+m.v[m.f[ind,1]]*t2+m.v[m.f[ind,2]]*t3
        ans.append(res)
    return np.array(ans)


points = []
voxels = []
nindex = []

fquery = []
pl = 1/32
i = 1/64
for t1 in range(32):
    j = 1/64
    for t2 in range(32):
        k=1/64
        for t3 in range(32):
            fquery.append([i,j,k])
            k+=pl
        j+=pl
    i+=pl
fquery = np.array(fquery)

cnt = 0
for f in flist:
    with open("data\\voxel\\"+f.replace("obj","binvox"),"rb") as voxf:
        voxel = read_as_3d_array(voxf)
        mesh = Mesh(path+f)
        mesh.v-=voxel.translate
        mesh.v/=voxel.scale
        mesh.write_obj(path.replace("mesh","mesh-trans")+f)
        tmp1 = sample(mesh,1000)
        tmp2 = sample(mesh,100000)
        mytree = scipy.spatial.cKDTree(tmp2)
        dis,ind = mytree.query(fquery)
        tmp3 = tmp2[ind].reshape([32,32,32,3])
        nindex.append(tmp3)
        points.append(tmp1)
        voxels.append(voxel.data)
    print(cnt)

    cnt += 1

points = np.array(points).astype(np.float32)
voxels = np.array(voxels)
nindex = np.array(nindex).astype(np.float32)
print(points.shape)
print(voxels.shape)
print(nindex.shape)
np.save("data\\points.npy",points)
np.save("data\\voxels.npy",voxels)
np.save("data\\nindex.npy",nindex)
