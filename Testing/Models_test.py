import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

DATASET_PATH = "../PlantVillage"
IMAGE_SIZE = 224
BATCH_SIZE = 64
TRAIN_SIZE = 0.8
VAL_SIZE = 0.1
TEST_SIZE = 0.1

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

model_directory_resnet50 = '../Models/ResNet50Model/1'
model_directory_mobilenetv2 = '../Models/MobileNetModel/2_first_top_acc'
model_directory_vgg19 = '../Models/vgg19Model/2_first_top_acc'

loaded_model_resnet50 = tf.keras.models.load_model(model_directory_resnet50)
loaded_model_mobilenetv2 = tf.keras.models.load_model(model_directory_mobilenetv2)
loaded_model_vgg19 = tf.keras.models.load_model(model_directory_vgg19)

print("VGG-19:")
scores_vgg19 = loaded_model_vgg19.evaluate(test_ds)
print("Loss: "+str((scores_vgg19[0]))+"; Accuracy: "+str((scores_vgg19[1])))
print("ResNet50:")
scores_resnet50 = loaded_model_resnet50.evaluate(test_ds)
print("Loss: "+str((scores_resnet50[0]))+"; Accuracy: "+str((scores_resnet50[1])))
print("MobileNetV2:")
scores_mobilenetv2 = loaded_model_mobilenetv2.evaluate(test_ds)
print("Loss: "+str((scores_mobilenetv2[0]))+"; Accuracy: "+str((scores_mobilenetv2[1])))
