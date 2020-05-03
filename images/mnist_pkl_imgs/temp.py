import pickle
import numpy as np
import keras
from keras.models import load_model

model = load_model('../mnist-net.h5')

fl = open("testim","w")
for i in range(1,101):
    with open(f'im{i}.pkl',"rb") as pf:
        data = pickle.load(pf)
    image = data[0]
    im = image.reshape(1,-1)
    out = model.predict(x=im,batch_size=1)
    for i in range(len(out)):
        fl.write(f'{out[i]},')
    fl.write('\n')

fl.close()

