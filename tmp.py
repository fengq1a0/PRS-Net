import numpy as np
a = np.load("a.npy")
b = np.load("b.npy")
for i in range(a.shape[0]):
    tmp1 = a[i]
    tmp2 = b[i]
    with open("data\\po\\"+str(i).zfill(3)+".obj",'w') as fi:
        for j in range(1000):
            fi.write('v %f %f %f 1 0 0\n' % (tmp1[j,0,0], tmp1[j,0,1], tmp1[j,0,2]))
        for j in range(1000):
            fi.write('v %f %f %f 0 1 0\n' % (tmp2[j,0,0], tmp2[j,0,1], tmp2[j,0,2]))
