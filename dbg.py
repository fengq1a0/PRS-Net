import tensorflow as tf
import numpy as np
from lib.PRSNet import PRSNet
from lib.util import rotate
import time

alpha = 0.3
epochs = 500
batch_size = 32
l_r = 0.01
w_r = 100

voxels = np.load("data\\voxels.npy")
voxels = voxels[:,:,:,:,np.newaxis]
voxels = voxels.astype(np.float32)   # [samples,32,32,32,1]

points = np.load("data\\points.npy")
points = points.astype(np.float32)   # [samples,1000,3]
nindex = np.load("data\\nindex.npy")
nindex = nindex.astype(np.float32)     # [samples,32,32,32]


dataset = tf.data.Dataset.from_tensor_slices((voxels,points,nindex)).shuffle(100).batch(32)
optimizer = tf.keras.optimizers.Adam(learning_rate=l_r)

def Qmul(t1,t2):
    t1 = tf.expand_dims(t1,3)
    t2 = tf.expand_dims(t2,4)
    tt0 = t1[:,:,:,:,0:1]
    tt1 = t1[:,:,:,:,1:2]
    tt2 = t1[:,:,:,:,2:3]
    tt3 = t1[:,:,:,:,3:4]
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
    np.save("a.npy",refy)

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

    ppp = tf.map_fn(lambda inp : tf.gather_nd(inp[0],inp[1]),(z,ind),dtype = tf.float32)
    np.save("b.npy",ppp[:,:,0:3,:])
    ppp = yy-ppp
    return tf.reduce_sum(tf.norm(ppp,axis = 3))/z.shape[0]


def loss2(para):
    t1 = para[:,0:3,0:3]
    t2 = para[:,3:6,1:4]
    t1 = tf.divide(t1,tf.expand_dims(tf.norm(t1,axis=2),2))
    t2 = tf.divide(t2,tf.expand_dims(tf.norm(t2,axis=2),2))
    t1 = tf.matmul(t1,tf.transpose(t1,[0,2,1]))
    t2 = tf.matmul(t2,tf.transpose(t2,[0,2,1]))
    I = np.eye(3)
    t1 = t1 - I
    t2 = t2 - I
    return tf.reduce_sum(tf.multiply(t1,t1))+tf.reduce_sum(tf.multiply(t2,t2))

model = PRSNet(alpha=0.3)
model.load_weights("checkpoints\\PSRNet075")
para = model(voxels)
loss1(para,points,nindex)