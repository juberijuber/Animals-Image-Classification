# %% [markdown]
# <a href="https://colab.research.google.com/github/juberijuber/Image-Classification/blob/main/Image%20Classification%20using_our_dataset.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %%
from keras.models import Sequential 
from keras.layers import Dense
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten

# %%

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# %% [markdown]
# 
# Image augementation techniques:
# 
# 
# 1.image rotation
# 2.image shifting
# 3.image scaling
# 4.image flipping
# 5.image noising
# 6.image blurring
# 
# 
# 

# %%
train_datagen=ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
test_datagen=ImageDataGenerator(rescale=1./255)

# %%
x_train=train_datagen.flow_from_directory(r"D:/Projects/Image-Classification/Datasets/testset",
target_size=(64,64),batch_size=32,class_mode="categorical")
x_test=test_datagen.flow_from_directory(r"D:/Projects/Image-Classification/Datasets/testset",
target_size=(64,64),batch_size=3249,class_mode="categorical")

# %%
print(x_train.class_indices)


# %%
model=Sequential()

# %%
#adding convolution layer(no.of filters,filter size,input shape,activation function)
model.add(Convolution2D(32,(3,3),input_shape=(64,64,3),activation="relu"))



# %%
#adding max pooling layer(pool_size)
model.add(MaxPooling2D(pool_size=(2,2)))


# %%
#input layer of ann
model.add(Flatten())
#add hidden layer(no.of neurons,activation=relu,weights)
model.add(Dense(units=128,activation="relu"))
#add output layer(no.of output classes=5,activation function=softmax)
model.add(Dense(units=5,activation="softmax"))


# %%
model.summary()


# %%
#configure the learning process(loss fucntion,accuracy,optimizer)
model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])

# %%
history=model.fit(x_train,steps_per_epoch=10,epochs=10,validation_data=x_test,validation_steps=20)


# %%
model.save("animal.h5")


# %%
from tensorflow.keras.models import load_model
from keras.preprocessing import image
import numpy as np

# %%
model=load_model("animal.h5")

# %%
img = image.load_img(r"D:\Projects\Image-Classification\Datasets\testset\bears\2Q__ (18).jpeg", target_size=(64, 64))

# %%
img


# %%
x=image.img_to_array(img)


# %%
x.shape

# %% [markdown]
# 1. Even if you are predicting a single image, the model still expects the input to be in a batch format
# 2. The shape of the input data for a Keras model should be (batch_size, height, width, channels).
# 

# %%
x=np.expand_dims(x,axis=0)

# %%
x.shape

# %%
y=model.predict(x)
pred=y.argmax( axis=1)


# %%
y


# %%
pred


# %%
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# %%
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

# %%
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer()
pred_multilabel = mlb.fit_transform(pred.reshape(-1, 1))  # Reshape if 'pred' is 1D
y_multilabel = mlb.transform(y.argmax(axis=1).reshape(-1, 1))  # Binarize y

# Now you can try calculating the accuracy score again:
accuracy = accuracy_score(y_multilabel, pred_multilabel)

print(accuracy)




# %%
x_train.class_indices

# %%
index=['bears', 'crows', 'elephants', 'racoons', 'rats']
result=str(index[pred[0]])

# %%
result


# %%


# %%


# %%


# %%


# %%



