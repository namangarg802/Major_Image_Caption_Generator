from keras.applications.resnet import ResNet50, preprocess_input, decode_predictions
from keras.utils import pad_sequences, load_img, img_to_array
from keras.preprocessing import image
from keras.models import load_model, Model
import matplotlib.pyplot as plt
import pickle
import numpy as np

import warnings
warnings.filterwarnings("ignore")


model = load_model("./model/naman_model.h5")
model.make_predict_function()

model_temp = ResNet50(weights="imagenet", input_shape=(224, 224, 3))

model_resnet = Model(model_temp.input, model_temp.layers[-2].output)
model_resnet.make_predict_function()



with open("./word_to_idx.pkl", "rb") as w2i:
    word_to_idx = pickle.load(w2i)

with open("./idx_to_word.pkl", "rb") as i2w:
    idx_to_word = pickle.load(i2w)


max_len = 35


def preprocess_image(img):
    img = load_img(img, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    #print((img))
    img = preprocess_input(img)
    return img


def encode_image(img):
    img = preprocess_image(img)
    #print((img))
    feature_vector = model_resnet.predict(img,verbose=0)
    #print((img))
    feature_vector = feature_vector.reshape(1, feature_vector.shape[1])
    
    return feature_vector


def predict_caption(photo):
    in_text = "startseq"

    for i in range(max_len):
        sequence = [word_to_idx[w]
                    for w in in_text.split() if w in word_to_idx]
        sequence = pad_sequences([sequence], maxlen=max_len, padding='post')

        ypred = model.predict([photo, sequence],verbose=0)
        ypred = ypred.argmax()
        word = idx_to_word[ypred]
        in_text += ' ' + word

        if word == 'endseq':
            break

    final_caption = in_text.split()
    final_caption = final_caption[1:-1]
    final_caption = ' '.join(final_caption)

    return final_caption


def caption_this_image(input_img):

    photo = encode_image(input_img)
    print(photo)
    caption = predict_caption(photo)

    return caption
