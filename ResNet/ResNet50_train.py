import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import Adam
import matplotlib.pyplot as plt

DATASET_PATH = "../PlantVillage"
IMAGE_SIZE = 224
IMAGE_SHAPE = (224, 224, 3)
INPUT_SHAPE = (32, 224, 224, 3)
BATCH_SIZE = 64
TRAIN_SIZE = 0.8
VAL_SIZE = 0.1
TEST_SIZE = 0.1
EPOCHS = 10

#Datensatz einlesen und Labels hinzufügen:
dataset = tf.keras.utils.image_dataset_from_directory(
 DATASET_PATH,
 shuffle=True,
 seed=123,
 image_size=(IMAGE_SIZE,IMAGE_SIZE),
 batch_size=BATCH_SIZE
 )
class_names = dataset.class_names
print(class_names)
n_classes = len(class_names)
print("Anzahl Klassen: "+str(n_classes))
print("Anzahl Batches: "+str(len(dataset)))

#train, test, validation split
#für train ds nehmen wir 80% der batches
train_size = int(len(dataset)*TRAIN_SIZE)
train_ds = dataset.take(train_size)
print("Es werden "+str(train_size)+" Batches zu Train hinzugefügt")
rest_ds = dataset.skip(train_size)
print("Es bleiben "+str(int(len(rest_ds)))+" Batches übrig")

#validation
val_size = int(len(dataset)*VAL_SIZE)
val_ds = rest_ds.take(val_size)
print("Es werden "+str(val_size)+" Batches zu Validation hinzugefügt")
rest_ds = rest_ds.skip(val_size)
print("Es bleiben "+str(int(len(rest_ds)))+" Batches übrig")

#test
test_size = int(len(dataset)*TEST_SIZE)
test_ds = rest_ds.take(test_size)
print("Es werden "+str(test_size)+" Batches zu Test hinzugefügt")

# vortrainiertes VGG19-Modell laden (ohne Klassifikationsschicht)
base_model = tf.keras.applications.resnet.ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Gewichtungen der vortrainierten Schichten einfrieren
for layer in base_model.layers:
 layer.trainable = False

base_model.summary()

resize_and_rescale = tf.keras.Sequential([
 tf.keras.layers.Resizing(IMAGE_SIZE, IMAGE_SIZE),
 tf.keras.layers.Rescaling(1.0/255)
])

# Klassifikationsschicht hinzufügen
inputs = tf.keras.Input(shape=IMAGE_SHAPE)
x = resize_and_rescale(inputs)
x = base_model(x)
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
predictions = Dense(n_classes, activation='softmax')(x) # 'num_classes' ist die Anzahl Ihrer Klassen

# Basismodell mit Klassifikationsschicht kombinieren
model = Model(inputs=inputs, outputs=predictions)
model.summary()



# Kompilieren des Modells
model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Trainieren des Modells
history = model.fit(train_ds, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=val_ds)

model_version=1
model.save(f"../Models/ResNet50Model/{model_version}")

#Training darstellen
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(range(EPOCHS), acc, label='Training Accuracy')
plt.plot(range(EPOCHS), val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(range(EPOCHS), loss, label='Training Loss')
plt.plot(range(EPOCHS), val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
