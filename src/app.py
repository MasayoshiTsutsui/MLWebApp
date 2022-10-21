from flask import Flask, render_template, request
from keras.models import load_model
from PIL import Image, ImageOps 
from keras.preprocessing.image import img_to_array 
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import Adam
from tensorflow.python.keras.backend import set_session # 
from keras.utils import np_utils
import tensorflow as tf #
graph = tf.get_default_graph() #
sess = tf.Session() #
set_session(sess) #

app = Flask(__name__)

model = Sequential()
model.add(Conv2D(16, (3, 3), padding='same',
            input_shape=(28, 28, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(10, activation='softmax'))

#5 損失関数・最適化関数・評価関数などを指定してモデルをコンパイル
model.compile(loss='categorical_crossentropy',
                optimizer=Adam(),
                metrics=['accuracy'])

@app.route('/')
def index():
    #(x_train, y_train), (x_test, y_test) = mnist.load_data()
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.files['input_file1'].stream
    label = request.form.get('input_label1')
    label = int(label)
    label = np_utils.to_categorical([label], 10)
    im = Image.open(data)

    #model = load_model('mnist_model_weight.h5') 

    img = im.resize((28,28))
    img = img.convert(mode='L')
    img = ImageOps.invert(img) 
    img = img_to_array(img)
    img = img.reshape(1, 28, 28, 1)
    img = img.astype('float32')/255

    with graph.as_default(): #
        set_session(sess) #
        history = model.ﬁt([img], label, batch_size=1, epochs=1,
                        verbose=0, validation_data=None)
        print(history)

    with graph.as_default(): #
        set_session(sess) #
        result = model.predict_classes(img)
    result = result[0]


    return render_template('result.html',result_output=result)

if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0", port=8000)