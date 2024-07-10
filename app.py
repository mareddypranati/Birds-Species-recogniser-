from __future__ import division, print_function

import os
from flask import Flask, flash, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
import cv2
import keras
import numpy as np
from keras.models import load_model

import warnings
warnings.filterwarnings("ignore")


#from __future__ import division, print_function
# coding=utf-8


# Flask utils
from flask import  render_template
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)




# from keras.applications.vgg16 import VGG16
# base_vgg16 = VGG16(weights='imagenet', include_top = False, input_shape=(150,150,3))
# from keras.layers import GlobalAveragePooling2D
# from keras.layers import Activation,Dense
# from keras.models import Sequential,load_model
# out = base_vgg16.output
# out = GlobalAveragePooling2D()(out)
# out = Dense(256, activation='relu')(out)
# out = Dense(256, activation='relu')(out)
# total_classes = 190
# predictions = Dense(190, activation='softmax')(out)

# from keras.models import Model
# model = Model(inputs=base_vgg16.input, outputs=predictions)
# for layer in base_vgg16.layers:
#     layer.trainable = False
# from keras.optimizers import Adam
# model.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy']) 


# model.save("last.h5")

# from tensorflow.keras.applications.resnet50 import ResNet50
# base_resnet = ResNet50(weights='imagenet', include_top = False, input_shape=(150,150,3))

# out = base_resnet.output
# out = GlobalAveragePooling2D()(out)
# out = Dense(256, activation='relu')(out)
# out = Dense(256, activation='relu')(out)
# total_classes = 190
# predictions = Dense(190, activation='softmax')(out)
# model = Model(inputs=base_resnet.input, outputs=predictions)

# for layer in base_resnet.layers:
#     layer.trainable = False
# model.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy']) 

# model.fit(
#         train_generator,
#         steps_per_epoch=train_generator.samples/train_generator.batch_size,
#         epochs=20,
#         validation_data=validation_generator,
#         validation_steps=validation_generator.samples/validation_generator.batch_size)

# model.save("last.h5")

model=load_model("last.h5")

model._make_predict_function()

print('Model loaded. Check http://127.0.0.1:5000/')
labels=['ALBATROSS',
'ALEXANDRINE PARAKEET',
'AMERICAN AVOCET',
'AMERICAN BITTERN',
'AMERICAN COOT',
'AMERICAN GOLDFINCH',
'AMERICAN KESTREL',
'AMERICAN PIPIT',
'AMERICAN REDSTART',
'ANHINGA',
'ANNAS HUMMINGBIRD',
'ANTBIRD',
'ARARIPE MANAKIN',
'BALD EAGLE',
'BALTIMORE ORIOLE',
'BANANAQUIT',
'BAR-TAILED GODWIT',
'BARN OWL',
'BARN SWALLOW',
'BAY-BREASTED WARBLER',
'BELTED KINGFISHER',
'BIRD OF PARADISE',
'BLACK FRANCOLIN',
'BLACK SKIMMER',
'BLACK SWAN',
'BLACK THROATED WARBLER',
'BLACK VULTURE',
'BLACK-CAPPED CHICKADEE',
'BLACK-NECKED GREBE',
'BLACK-THROATED SPARROW',
'BLACKBURNIAM WARBLER',
'BLUE GROUSE',
'BLUE HERON',
'BOBOLINK',
'BROWN NOODY',
'BROWN THRASHER',
'CACTUS WREN',
'CALIFORNIA CONDOR',
'CALIFORNIA GULL',
'CALIFORNIA QUAIL',
'CANARY',
'CAPE MAY WARBLER',
'CARMINE BEE-EATER',
'CASPIAN TERN',
'CASSOWARY',
'CHARA DE COLLAR',
'CHIPPING SPARROW',
'CINNAMON TEAL',
'COCK OF THE  ROCK',
'COCKATOO',
'COMMON GRACKLE',
'COMMON HOUSE MARTIN',
'COMMON LOON',
'COMMON POORWILL',
'COMMON STARLING',
'COUCHS KINGBIRD',
'CRESTED AUKLET',
'CRESTED CARACARA',
'CROW',
'CROWNED PIGEON',
'CUBAN TODY',
'CURL CRESTED ARACURI',
'D-ARNAUDS BARBET',
'DARK EYED JUNCO',
'DOWNY WOODPECKER',
'EASTERN BLUEBIRD',
'EASTERN MEADOWLARK',
'EASTERN ROSELLA',
'EASTERN TOWEE',
'ELEGANT TROGON',
'ELLIOTS  PHEASANT',
'EMPEROR PENGUIN',
'EMU',
'EURASIAN MAGPIE',
'EVENING GROSBEAK',
'FLAME TANAGER',
'FLAMINGO',
'FRIGATE',
'GILA WOODPECKER',
'GLOSSY IBIS',
'GOLD WING WARBLER',
'GOLDEN CHLOROPHONIA',
'GOLDEN EAGLE',
'GOLDEN PHEASANT',
'GOULDIAN FINCH',
'GRAY CATBIRD',
'GRAY PARTRIDGE',
'GREEN JAY',
'GREY PLOVER',
'GUINEAFOWL',
'HAWAIIAN GOOSE',
'HOODED MERGANSER',
'HOOPOES',
'HORNBILL',
'HOUSE FINCH',
'HOUSE SPARROW',
'HYACINTH MACAW',
'IMPERIAL SHAQ',
'INCA TERN',
'INDIGO BUNTING',
'JABIRU',
'JAVAN MAGPIE',
'KILLDEAR',
'KING VULTURE',
'LARK BUNTING',
'LILAC ROLLER',
'LONG-EARED OWL',
'MALEO',
'MALLARD DUCK',
'MANDRIN DUCK',
'MARABOU STORK',
'MASKED BOOBY',
'MIKADO  PHEASANT',
'MOURNING DOVE',
'MYNA',
'NICOBAR PIGEON',
'NORTHERN CARDINAL',
'NORTHERN FLICKER',
'NORTHERN GANNET',
'NORTHERN GOSHAWK',
'NORTHERN JACANA',
'NORTHERN MOCKINGBIRD',
'NORTHERN RED BISHOP',
'OCELLATED TURKEY',
'OSPREY',
'OSTRICH',
'PAINTED BUNTIG',
'PARADISE TANAGER',
'PARUS MAJOR',
'PEACOCK',
'PELICAN',
'PEREGRINE FALCON',
'PINK ROBIN',
'PUFFIN',
'PURPLE FINCH',
'PURPLE GALLINULE',
'PURPLE MARTIN',
'PURPLE SWAMPHEN',
'QUETZAL',
'RAINBOW LORIKEET',
'RAZORBILL',
'RED BISHOP WEAVER',
'RED FACED CORMORANT',
'RED HEADED DUCK',
'RED HEADED WOODPECKER',
'RED HONEY CREEPER',
'RED THROATED BEE EATER',
'RED WINGED BLACKBIRD',
'RED WISKERED BULBUL',
'RING-BILLED GULL',
'RING-NECKED PHEASANT',
'ROADRUNNER',
'ROBIN',
'ROCK DOVE',
'ROSY FACED LOVEBIRD',
'ROUGH LEG BUZZARD',
'RUBY THROATED HUMMINGBIRD',
'RUFOUS KINGFISHER',
'RUFUOS MOTMOT',
'SAND MARTIN',
'SCARLET IBIS',
'SCARLET MACAW',
'SHOEBILL',
'SNOWY EGRET',
'SNOWY OWL',
'SORA',
'SPANGLED COTINGA',
'SPLENDID WREN',
'SPOONBILL',
'STEAMER DUCK',
'STORK BILLED KINGFISHER',
'STRAWBERRY FINCH',
'TAIWAN MAGPIE',
'TEAL DUCK',
'TIT MOUSE',
'TOUCHAN',
'TRUMPTER SWAN',
'TURKEY VULTURE',
'TURQUOISE MOTMOT',
'VARIED THRUSH',
'VENEZUELIAN TROUPIAL',
'VERMILION FLYCATHER',
'VIOLET GREEN SWALLOW',
'WHITE CHEEKED TURACO',
'WHITE NECKED RAVEN',
'WHITE TAILED TROPIC',
'WILD TURKEY',
'WILSONS BIRD OF PARADISE',
'WOOD DUCK',
'YELLOW HEADED BLACKBIRD']


def model_predict(img_path, model):
    img = cv2.imread(img_path)
    img = cv2.resize(img,(150,150))
    img = np.reshape(img,[1,150,150,3])
    res = model.predict(img).argmax(axis=-1)
    result=labels[int(res)]
    return result

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        result = model_predict(file_path, model)
        
        return result
    return None

if __name__ == '__main__':
    app.run(debug=False, threaded=False)                            #for deploying in local system
    #app.run(debug=False,threaded=False,host='0.0.0.0',port=8080)      #for deploying in aws
    
    
