import numpy as np
import os
from MultiResUNet3D import MultiResUnet3D
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint 

def dice_coef(y_true, y_pred):
    smooth = 0.0
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def jacard(y_true, y_pred):

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum ( y_true_f * y_pred_f)
    union = K.sum ( y_true_f + y_pred_f - y_true_f * y_pred_f)

    return intersection/union

data = np.load('training_data.npz')

X_train = data['X_train']
X_test = data['X_test']
Y_train = data['Y_train']
Y_test = data['Y_test']

batchSize = 16
n_epochs = 100

print('[INFO] Train set shapes are:')
print(X_train.shape)
print(Y_train.shape)
print('[INFO] Test set shapes are:')
print(X_test.shape)
print(Y_test.shape)

model = MultiResUnet3D(16, 64, 64, 1)

checkpoint = ModelCheckpoint('weights.hdf5', monitor="val_loss",
save_best_only=True, verbose=1)
callbacks = [checkpoint]


#model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[dice_coef, jacard, 'accuracy'])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

H = model.fit(x=X_train, y=Y_train, batch_size=batchSize, epochs=n_epochs, verbose=1,
             validation_data=(X_test,Y_test), callbacks=callbacks)

print('[INFO] Finished training')
print(H.history.keys())

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, n_epochs), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, n_epochs), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, n_epochs), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, n_epochs), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig('loss_accuracy.png')

# plt.figure()
# plt.plot(np.arange(0, n_epochs), H.history["jacard"], label="jacard")
# plt.plot(np.arange(0, n_epochs), H.history["val_jacard"], label="val_jacard")
# plt.title("Jackard")
# plt.xlabel("Epoch #")
# plt.ylabel("Loss/Accuracy")
# plt.legend()
# plt.savefig('jackard_loss_accuracy.png')