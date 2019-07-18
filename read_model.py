import keras
from keras.models import load_model

json_file = open(r'''seventh_expirience.json''')
loaded_model_json = json_file.read()
json_file.close()
try:
	model = keras.models.model_from_json(loaded_model_json)
finally:
	print("")

model = keras.models.model_from_json(loaded_model_json)
model.load_weights(r'''seventh_expirience_weights.h5''')
model.compile(loss = 'categorical_crossentropy', optimizer = 'Adam', metrics=['accuracy'])
test_dir = r'''predict'''
from keras.preprocessing.image import ImageDataGenerator
test_data = ImageDataGenerator(rescale = 1. / 255)
predict = test_data.flow_from_directory(test_dir, target_size=(150, 150), color_mode="rgb", shuffle = False, class_mode='categorical', batch_size=1)
filenames = predict.filenames

nb_samples = len(filenames)
predict = model.predict_generator(predict,steps = nb_samples)

print("Prediction was passed with: ")
print(str(predict[0][0]*100) + "% - it is with smile")
print(str(predict[0][1]*100) + "% - it is without smile")
input()