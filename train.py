import tensorflow as tf
import numpy as np
from lib.PSRNet import PRSNet
from lib.util import rotate

num_epochs = 5
alpha = 0.05
lr = 0.01
omega = 100

def ref(para,points,nindex):
    res = 0
    for i in range(points.shape[0]):
        tmp = points[i]-2*para[0:3]*(tf.reduce_sum(tf.multiply(points[i],para[0:3]))+para[3]) / tf.reduce_sum(tf.multiply(para[0:3],para[0:3]))
        ind = tmp * 32
        ind = tf.floor(ind)
        ind = tf.minimum(ind,np.array([31,31,31]))
        ind = tf.maximum(ind,np.array([0,0,0]))
        ind = tf.cast(ind,dtype=tf.int32)
        res += tf.norm(tmp-points[nindex[ind[0],ind[1],ind[2]]])
    return res

def rot(para,points,nindex):
    res = 0
    for i in range(points.shape[0]):
        tmp = rotate(para,points[i])
        ind = tmp * 32
        ind = tf.floor(ind)
        ind = tf.minimum(ind,np.array([31,31,31]))
        ind = tf.maximum(ind,np.array([0,0,0]))
        ind = tf.cast(ind,dtype=tf.int32)
        res += tf.norm(tmp-points[nindex[ind[0],ind[1],ind[2]]])
    return res


def loss1(para,points,nindex):
    res = 0
    for i in range(para.shape[0]):
        for j in range(para.shape[1]):
            res += ref(para[i,j],points[i],nindex[i])
        '''
        for j in range(3,6,1):
            res += rot(para[i,j],points[i],nindex[i])
        '''
    return res/para.shape[0]

def loss2(para):
    ans = 0
    for i in range(para.shape[0]):
        n11 = para[i,0,0:3]/tf.norm(para[i,0,0:3])
        n12 = para[i,1,0:3]/tf.norm(para[i,1,0:3])
        n13 = para[i,2,0:3]/tf.norm(para[i,2,0:3])
        '''
        n21 = para[i,3,1:4]/tf.norm(para[i,3,1:4])
        n22 = para[i,4,1:4]/tf.norm(para[i,4,1:4])
        n23 = para[i,5,1:4]/tf.norm(para[i,5,1:4])
        '''
        M1 = tf.stack([n11,n12,n13])
        #M2 = tf.stack([n21,n22,n23])
        
        III = np.eye(3)
        A = tf.matmul(M1,tf.transpose(M1))-III
        #B = tf.matmul(M2,tf.transpose(M2))-III
        ans += (tf.reduce_sum(tf.multiply(A,A)))# + tf.reduce_sum(tf.multiply(B,B)))
    return ans/para.shape[0]

def grad(model,x,y,z):
    with tf.GradientTape() as tape:
        para = model(x)
        loss = omega*loss2(para) + loss1(para,y,z)
    return loss,tape.gradient(loss, model.trainable_variables)


voxels = np.load("data\\voxels.npy")
voxels = voxels[:,:,:,:,np.newaxis]
voxels = voxels.astype(np.float32)   # [samples,32,32,32,1]
points = np.load("data\\points.npy")
points = points.astype(np.float32)   # [samples,1000,3]
nindex = np.load("data\\nindex.npy")
nindex = nindex.astype(np.int32)     # [samples,32,32,32]

ds = tf.data.Dataset.from_tensor_slices((voxels,points,nindex)).shuffle(100).batch(32)
model = PRSNet(alpha=alpha)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

for epoch in range(num_epochs):
    cnt = 0
    for x,y,z in ds:
        loss,grads = grad(model,x,y,z)
        print("epoch: ",epoch," batch: ",cnt," loss: ",loss)
        cnt+=1
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
    model.save_weights("checkpoints\\PSRNet"+str(epoch))