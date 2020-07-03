import os
path = "data\\mesh\\"
flist = sorted(os.listdir(path))
for f in flist:
    os.system("data\\binvox.exe -d 32 "+path+f+" && move "
            +path+f.replace("obj","binvox")+" data\\voxel\\"+f.replace("obj","binvox"))