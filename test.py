from lib.util import visualize
import numpy as np
from lib.PRSNet import PRSNet
import tensorflow as tf

voxels = np.load("data\\voxels.npy")
voxels = voxels[:,:,:,:,np.newaxis]
voxels = voxels.astype(np.float32)   # [samples,32,32,32,1]
points = np.load("data\\points.npy")
points = points.astype(np.float32)   # [samples,1000,3]
nindex = np.load("data\\nindex.npy")
nindex = nindex.astype(np.float32)     # [samples,32,32,32]

def Qmul(t1,t2):
    t1 = tf.expand_dims(t1,3)
    t2 = tf.expand_dims(t2,4)
    tt0 = tf.expand_dims(t1[:,:,:,:,0],4)
    tt1 = tf.expand_dims(t1[:,:,:,:,1],4)
    tt2 = tf.expand_dims(t1[:,:,:,:,2],4)
    tt3 = tf.expand_dims(t1[:,:,:,:,3],4)
    rr0 = tf.matmul(tf.concat([tt0,tt1*-1,tt2*-1,tt3*-1],axis=4),t2)
    rr1 = tf.matmul(tf.concat([tt1,tt0,tt3*-1,tt2],axis=4),t2)
    rr2 = tf.matmul(tf.concat([tt2,tt3,tt0,tt1*-1],axis=4),t2)
    rr3 = tf.matmul(tf.concat([tt3,tt2*-1,tt1,tt0],axis=4),t2)
    res = tf.concat([rr0,rr1,rr2,rr3],axis=4)
    return tf.reshape(res,[res.shape[0],res.shape[1],res.shape[2],4])

def loss1(para,y,z):
    refy = np.ones([y.shape[0],y.shape[1],1]).astype(np.float32)
    refy = tf.concat([y,refy],axis=2)
    refy = tf.expand_dims(refy,3) # b,1000,4,1
    refpara = tf.expand_dims(para[:,0:3,:],1) # b,1,3,4
    refpara = tf.matmul(refpara,refy) # b,1000,3,1
    refy = tf.expand_dims(para[:,0:3,0:3],1) # b,1,3,3
    refy = tf.expand_dims(refy,4) #b,1,3,3,1
    refy = tf.matmul(tf.transpose(refy,[0,1,2,4,3]),refy) # b,1,3,1,1
    refpara = tf.divide(refpara,tf.squeeze(refy,4)) # b,1000,3,1
    refpara = tf.multiply(refpara,tf.expand_dims(para[:,0:3,0:3],1))*2 # b,1000,3,3
    refy = tf.expand_dims(y,2)-refpara # b,1000,3,3

    roty = np.zeros([y.shape[0],y.shape[1],1]).astype(np.float32)
    roty = tf.concat([roty,y],axis=2)
    roty = tf.expand_dims(roty,2) # b,1000,1,4
    rotpara = tf.expand_dims(para[:,3:6,:],1) # b,1,3,4
    rotpara = rotpara/tf.expand_dims(tf.norm(rotpara,axis=3),3)
    rotpara_inv = tf.multiply(rotpara,np.array([1,-1,-1,-1])) # b,1,3,4
    roty = Qmul(Qmul(rotpara,roty),rotpara_inv)[:,:,:,1:4] # b,1000,3,3
    
    yy = tf.concat([refy,roty],axis=2)
    ind = yy * 32
    ind = tf.clip_by_value(ind,0.5,31.5)
    ind = tf.floor(ind)
    ind = tf.cast(ind,dtype=tf.int32)

    np.save("res.npy",yy)

    ppp = tf.map_fn(lambda inp : tf.gather_nd(inp[0],inp[1]),(z,ind),dtype = tf.float32)
    ppp = yy-ppp
    return tf.reduce_sum(tf.norm(ppp,axis = 3))

model = PRSNet(alpha=0.3)
model.load_weights("checkpoints\\PSRNet499")
res = model(voxels)
loss1(res,points,nindex)