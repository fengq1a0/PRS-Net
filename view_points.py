import numpy as np
a = np.load("data\\nindex.npy")
for i in range(a.shape[0]):
    tmp = a[i].reshape([32*32*32,3])
    with open("data\\point\\"+str(i).zfill(3)+".obj",'w') as fi:
        for j in range(32*32*32):
            fi.write('v %f %f %f 1 0 0\n' % (tmp[j,0], tmp[j,1], tmp[j,2]))