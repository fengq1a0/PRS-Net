import tensorflow as tf
class PRSNet(tf.keras.Model):
    def __init__(self, alpha):
        super(PRSNet, self).__init__()
        self.cov = tf.keras.Sequential()
        self.cov.add(tf.keras.layers.Conv3D(4,(3,3,3),padding="same"))
        self.cov.add(tf.keras.layers.MaxPool3D())
        self.cov.add(tf.keras.layers.LeakyReLU(alpha))
        self.cov.add(tf.keras.layers.Conv3D(8,(3,3,3),padding="same"))
        self.cov.add(tf.keras.layers.MaxPool3D())
        self.cov.add(tf.keras.layers.LeakyReLU(alpha))
        self.cov.add(tf.keras.layers.Conv3D(16,(3,3,3),padding="same"))
        self.cov.add(tf.keras.layers.MaxPool3D())
        self.cov.add(tf.keras.layers.LeakyReLU(alpha))
        self.cov.add(tf.keras.layers.Conv3D(32,(3,3,3),padding="same"))
        self.cov.add(tf.keras.layers.MaxPool3D())
        self.cov.add(tf.keras.layers.LeakyReLU(alpha))
        self.cov.add(tf.keras.layers.Conv3D(64,(3,3,3),padding="same"))
        self.cov.add(tf.keras.layers.MaxPool3D())
        self.cov.add(tf.keras.layers.LeakyReLU(alpha))
        self.cov.add(tf.keras.layers.Flatten())
        
        self.fc1 = tf.keras.Sequential()
        self.fc1.add(tf.keras.layers.Dense(32))
        self.fc1.add(tf.keras.layers.LeakyReLU(alpha))
        self.fc1.add(tf.keras.layers.Dense(16))
        self.fc1.add(tf.keras.layers.LeakyReLU(alpha))
        self.fc1.add(tf.keras.layers.Dense(4))
        self.fc1.add(tf.keras.layers.LeakyReLU(alpha))
        
        self.fc2 = tf.keras.Sequential()
        self.fc2.add(tf.keras.layers.Dense(32))
        self.fc2.add(tf.keras.layers.LeakyReLU(alpha))
        self.fc2.add(tf.keras.layers.Dense(16))
        self.fc2.add(tf.keras.layers.LeakyReLU(alpha))
        self.fc2.add(tf.keras.layers.Dense(4))
        self.fc2.add(tf.keras.layers.LeakyReLU(alpha))
        
        self.fc3 = tf.keras.Sequential()
        self.fc3.add(tf.keras.layers.Dense(32))
        self.fc3.add(tf.keras.layers.LeakyReLU(alpha))
        self.fc3.add(tf.keras.layers.Dense(16))
        self.fc3.add(tf.keras.layers.LeakyReLU(alpha))
        self.fc3.add(tf.keras.layers.Dense(4))
        self.fc3.add(tf.keras.layers.LeakyReLU(alpha))
        '''
        self.fc4 = tf.keras.Sequential()
        self.fc4.add(tf.keras.layers.Dense(32))
        self.fc4.add(tf.keras.layers.LeakyReLU(alpha))
        self.fc4.add(tf.keras.layers.Dense(16))
        self.fc4.add(tf.keras.layers.LeakyReLU(alpha))
        self.fc4.add(tf.keras.layers.Dense(4))
        self.fc4.add(tf.keras.layers.LeakyReLU(alpha))
        
        self.fc5 = tf.keras.Sequential()
        self.fc5.add(tf.keras.layers.Dense(32))
        self.fc5.add(tf.keras.layers.LeakyReLU(alpha))
        self.fc5.add(tf.keras.layers.Dense(16))
        self.fc5.add(tf.keras.layers.LeakyReLU(alpha))
        self.fc5.add(tf.keras.layers.Dense(4))
        self.fc5.add(tf.keras.layers.LeakyReLU(alpha))
        
        self.fc6 = tf.keras.Sequential()
        self.fc6.add(tf.keras.layers.Dense(32))
        self.fc6.add(tf.keras.layers.LeakyReLU(alpha))
        self.fc6.add(tf.keras.layers.Dense(16))
        self.fc6.add(tf.keras.layers.LeakyReLU(alpha))
        self.fc6.add(tf.keras.layers.Dense(4))
        self.fc6.add(tf.keras.layers.LeakyReLU(alpha))
        '''
        

    def call(self, x):
        feature = self.cov(x)

        return tf.concat([tf.expand_dims(self.fc1(feature),1),
                        tf.expand_dims(self.fc2(feature),1),
                        tf.expand_dims(self.fc3(feature),1)],axis=1)

'''
,
                        tf.expand_dims(self.fc4(feature),1),
                        tf.expand_dims(self.fc5(feature),1),
                        tf.expand_dims(self.fc6(feature),1)
'''