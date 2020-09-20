#importing libraries
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten,Dropout
from keras.layers import Dense
import tensorflow as tf
import numpy as np
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials
from keras.preprocessing.image import ImageDataGenerator


#acsess to my drive
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)


#degumentaion the pictures and upload them from my drive
train_datagen = ImageDataGenerator(rescale = 1./255,
                                        rotation_range=20,
                                        shear_range = 0.2,
                                        zoom_range = 0.2,
                                        width_shift_range=0.2,
                                        height_shift_range=0.2,
                                        fill_mode='nearest', 
                                        horizontal_flip = True)
        
test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('/content/drive/My Drive/train (1)',
                                                        target_size = (64, 64),
                                                        class_mode = 'categorical')
test_set = test_datagen.flow_from_directory('/content/drive/My Drive/test (1)',
                                                   target_size = (64, 64),
                                                  class_mode = 'categorical')



#Bulding the model with 2  hidden layer 
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2))
classifier.add(Convolution2D(64, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

#flattern the data               
classifier.add(Flatten())

#output layers              
classifier.add(Dense(128, activation = 'relu'))
classifier.add(Dense(256, activation = 'relu'))
classifier.add(Dense(5, activation = 'softmax'))

#compile the modlel
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#early stopping to avoid overfitting
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

#fit the project  
classifier.fit_generator(training_set,
      steps_per_epoch=32, epochs=50,callbacks=[callback],validation_data=test_set)

#predict the test set 
h  = classifier.predict(test_set)
h = np.argmax(h,axis=1)

# getting the real predict               
ty=[]
for j in range(29):
     y = test_set[j][1]
     for i in y :
            i = list(i)
            ty.append(i.index(1))
              
ty=np.array(ty)

#creating function to measure the accuracy
def acc(y_true, y_pred):
          return np.equal(y_test,y_pred).mean()
print("accuracy: " + str(acc(h, ty)))


#classify each number toits string    
yy=[]
for i in yt:
         if i == 0 :
            yy.append("elefante")
         elif i == 1 :
           yy.append("farfalla")
         elif i == 2 :
           yy.append("mucca")
         elif  i == 3 :
            yy.append("pecora")
         else :
           yy.append("scoiattolo")
  
