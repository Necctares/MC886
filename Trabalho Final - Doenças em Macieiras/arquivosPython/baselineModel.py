'''Modelo BASELINE'''

modelo = tf.keras.models.Sequential()


modelo.add(tf.keras.layers.Conv2D(32,3,input_shape=(255,255,3)))
modelo.add(tf.keras.layers.LeakyReLU())
modelo.add(tf.keras.layers.Conv2D(16,5))
modelo.add(tf.keras.layers.LeakyReLU())
modelo.add(tf.keras.layers.MaxPooling2D())
modelo.add(tf.keras.layers.Dropout(0.5))
modelo.add(tf.keras.layers.BatchNormalization())

modelo.add(tf.keras.layers.Flatten())
modelo.add(tf.keras.layers.Dense(16, activation='relu'))
modelo.add(tf.keras.layers.Dropout(0.5))
modelo.add(tf.keras.layers.BatchNormalization())
modelo.add(tf.keras.layers.Dense(4, activation='softmax'))

modelo.summary()  
