from train import rfc
import numpy as np

def __predict(weight, height, width, depth):
        X = np.array(weight, height, width, depth)
        prediction = rfc.predict(X)
        print(prediction)