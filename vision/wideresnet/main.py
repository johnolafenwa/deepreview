import keras
from keras.layers import *
from keras.optimizers import *
from keras.datasets import *
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
import os
from keras.callbacks import *
from math import ceil


# A single wide resnet module consisting of two 3 x 3 convs
def wide_resnet_module(x, filters, stride=1,dropout_ratio=0.0):
    res = x
    if x.shape[3] != filters:
        res = Conv2D(filters, kernel_size=1, strides=stride, padding="same",kernel_initializer="he_normal")(res)

    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(filters, kernel_size=3, strides=stride, padding="same",kernel_initializer="he_normal")(x)

    x = Dropout(dropout_ratio)(x)

    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(filters, kernel_size=3, strides=1, padding="same",kernel_initializer="he_normal")(x)

    x = add([x, res])

    return x

# A wide resnet block consisting of N number of wide resnet modules
def wide_resnet_block(x, filters, num_layers, stride=1,dropout_ratio=0.0):

    x = wide_resnet_module(x,filters=filters,stride=stride,dropout_ratio=dropout_ratio)
    for i in range(num_layers - 1):
        x = wide_resnet_module(x, filters=filters,dropout_ratio=dropout_ratio)
    return x


# The Wide Resnet model consisting of three blocks of conv, each block contains n modules. Each module contains 2 conv layers
#K is the widening factor that specifies how much to enlarge the filter sizes
def Resnet(input_shape, num_layers=16, k=4, num_classes=10,dropout_ratio=0.25):
    if not (num_layers - 4) % 6 == 0:
        raise ValueError("Num Layers must be 6n + 4 where n is number of modules per block")

    n = int((num_layers - 4) / 6)

    input = Input(input_shape)
    x = Conv2D(16,kernel_size=3,padding="same")(input)

    x = wide_resnet_block(x,filters=int(16 * k),num_layers=n,dropout_ratio=dropout_ratio)
    x = wide_resnet_block(x, filters=int(32 * k), num_layers=n,stride=2,dropout_ratio=dropout_ratio)
    x = wide_resnet_block(x, filters=int(64 * k), num_layers=n,stride=2,dropout_ratio=dropout_ratio)

    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = GlobalAvgPool2D()(x)

    x = Dense(num_classes)(x)
    x = Activation("softmax")(x)

    model = Model(inputs=input, outputs=x, name="WideResnet {} - {}".format(num_layers,k))

    return model



# *********************TRAINING CODE FOR CIFAR10 *****************#

# Load the CIFAR10 data.
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Get the input image dimensions
input_shape = x_train.shape[1:]

# Specify batch size and number of classes
num_classes = 10
batch_size = 128


model = Resnet(input_shape=(32, 32, 3), num_layers=16,k=4,num_classes=num_classes)
model.summary()

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Normalize data by subtracting mean and dividing by std.
x_train = x_train.astype('float32')
x_train = (x_train - x_train.mean(axis=0)) / (x_train.std(axis=0))
x_test = x_test.astype('float32')
x_test = (x_test - x_test.mean(axis=0)) / (x_test.std(axis=0))

num_train_samples = x_train.shape[0]
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print('y_train shape:', y_train.shape)

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Define path to save models
model_direc = os.path.join(os.getcwd(), 'cifar10_saved_modelsbest')

model_name = 'cifar10_model.{epoch:03d}.h5'
if not os.path.isdir(model_direc):
    os.makedirs(model_direc)

modelpath = os.path.join(model_direc, model_name)

# Prepare callbacks for saving models
checkpoint = ModelCheckpoint(filepath=modelpath,
                             monitor='val_acc',
                             verbose=1,
                             save_weights_only=True,
                             save_best_only=True)


# Define model schedule
def lr_schedule(epoch):
    """Learning rate is set to reduce by a factor of 20 at epochs 60, 120 and 160
    """
    lr = 0.1

    if epoch > 160:
      lr = lr * (0.2 ** 3)
    elif epoch > 120:
      lr = lr * (0.2 ** 2)
    elif epoch > 60:
      lr = lr * 0.2

    print('Learning rate: ', lr)

    return lr



# Compile the model to use SGD optimizer and categorical cross entropy loss
model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=lr_schedule(0), momentum=0.90, nesterov=True),
              metrics=['accuracy'])

# Create the learning rate scheduler
lr_scheduler = LearningRateScheduler(lr_schedule)

# Define callbacks
callbacks = [checkpoint, lr_scheduler]

# preprocessing and realtime data augmentation:
datagen = ImageDataGenerator(rotation_range=10,
                             width_shift_range=5. / 32,
                             height_shift_range=5. / 32,
                             horizontal_flip=True)

# Compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied).
datagen.fit(x_train)

# Define number of epochs and number of steps per epoch
epochs = 200
steps_per_epoch = ceil(num_train_samples / batch_size)

# Fit the model on the batches generated by datagen.flow().
model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                    validation_data=(x_test, y_test),
                    epochs=epochs, steps_per_epoch=steps_per_epoch, verbose=1, workers=4,
                    callbacks=callbacks)

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
