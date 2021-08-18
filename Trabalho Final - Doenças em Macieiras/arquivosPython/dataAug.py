from PIL import Image
tt = np.empty((1,255,255,3))
tt[0] = np.uint8(Image.open(img_dir + f'Train_{0}.jpg').resize((255, 255)))
data_aug = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=15,zoom_range=0.2,width_shift_range=0.2,height_shift_range=0.2,horizontal_flip=True,vertical_flip=True)
img = data_aug.flow(tt,batch_size=1)