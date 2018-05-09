import keras
from keras.layers import *
from keras.optimizers import *
from keras.datasets import *
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
import os
from keras.callbacks import *
from math import ceil


# A single resnet module consisting of 1 x 1 conv - 3 x 3 conv and 1 x 1 conv
def resnet_module(x, filters, pool=False):
    res = x
    stride = 1
    if pool:
        stride = 2
        res = Conv2D(filters, kernel_size=1, strides=2, padding="same")(res)
        res = BatchNormalization()(res)

    x = Conv2D(int(filters / 4), kernel_size=1, strides=stride, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(int(filters / 4), kernel_size=3, strides=1, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(filters, kernel_size=1, strides=1, padding="same")(x)
    x = BatchNormalization()(x)

    x = add([x, res])
    x = Activation("relu")(x)

    return x


def resnet_first_module(x, filters):
    res = x
    stride = 1
    res = Conv2D(filters, kernel_size=1, strides=1, padding="same")(res)
    res = BatchNormalization()(res)

    x = Conv2D(int(filters / 4), kernel_size=1, strides=stride, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(int(filters / 4), kernel_size=3, strides=1, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(filters, kernel_size=1, strides=1, padding="same")(x)
    x = BatchNormalization()(x)

    x = add([x, res])
    x = Activation("relu")(x)

    return x


# A resnet block consisting of N number of resnet modules, first layer has is pooled.
def resnet_block(x, filters, num_layers, pool_first_layer=True):
    for i in range(num_layers):
        pool = False
        if i == 0 and pool_first_layer: pool = True
        x = resnet_module(x, filters=filters, pool=pool)
    return x


# The Resnet model consisting of Conv - block1 - block2 - block3 - block 4 - FC with Softmax
def Resnet(input_shape, num_layers=50, num_classes=10):
    if num_layers not in [50, 101, 152]:
        raise ValueError("Num Layers must be either 50, 101 or 152")

    block_layers = {50: [3, 4, 6, 3],
                    101: [3, 4, 23, 3],
                    152: [3, 8, 36, 3]
                    }
    block_filters = {50: [256, 512, 1024, 2048],
                     101: [256, 512, 1024, 2048],
                     152: [256, 512, 1024, 2048]
                     }
    layers = block_layers[num_layers]
    filters = block_filters[num_layers]
    input = Input(input_shape)

    x = Conv2D(64, kernel_size=7, strides=2, padding="same")(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    x = resnet_first_module(x, filters[0])

    for i in range(4):
        num_filters = filters[i]
        num_layers = layers[i]

        pool_first = True
        if i == 0:
            pool_first = False
            num_layers = num_layers - 1
        x = resnet_block(x, filters=num_filters, num_layers=num_layers, pool_first_layer=pool_first)

    x = GlobalAveragePooling2D()(x)
    x = Dense(num_classes)(x)
    x = Activation("softmax")(x)

    model = Model(inputs=input, outputs=x, name="Resnet{}".format(num_layers))

    return model


model = Resnet(input_shape=(32, 32, 3), num_layers=50,num_classes=1000)
model.summary()

# Load the CIFAR10 data.
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Get the input image dimensions
input_shape = x_train.shape[1:]

# Specify batch size and number of classes
num_classes = 10
batch_size = 128

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
    """Learning rate is set to reduce by a factor of 10 after every 30 epochs
    """
    lr = 0.001

    if epoch % 30 == 0 and epoch > 0:
        lr *= 0.1 ** (epoch / 30)

    print('Learning rate: ', lr)

    return lr


# *********************TRAINING CODE FOR CIFAR10 *****************#

# Compile the model to use SGD optimizer and categorical cross entropy loss
model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=lr_schedule(0), momentum=0.95, nesterov=True),
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
