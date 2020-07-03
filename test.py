from lib.util import visualize
import numpy as np
from lib.PRSNet import PRSNet

voxels = np.load("data\\voxels.npy")
voxels = voxels[:,:,:,:,np.newaxis]
voxels = voxels.astype(np.float32)   # [samples,32,32,32,1]
points = np.load("data\\points.npy")
points = points.astype(np.float32)   # [samples,1000,3]

model = PRSNet(alpha=0.3)
model.load_weights("checkpoints\\PSRNet499")
res = model(voxels)
print(res.shape)
for i in range(points.shape[0]):
    visualize(res[i,0],points[i]*32,"result\\"+str(i).zfill(3)+"_0.png")
    visualize(res[i,1],points[i]*32,"result\\"+str(i).zfill(3)+"_1.png")
    visualize(res[i,2],points[i]*32,"result\\"+str(i).zfill(3)+"_2.png")