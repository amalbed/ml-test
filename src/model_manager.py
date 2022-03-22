import json
import os
import numpy as np


from .feature_extractor import fingerprint_features


from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

from tensorflow.keras.models import Sequential


import tensorflow as tf
from sklearn.metrics import accuracy_score
from tensorflow import expand_dims

from rdkit.Chem import DataStructs


class molecule:

    def create_baseline():
        model = Sequential()
        model.add(Dense(2048, input_dim=2048, activation="relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(1024, activation="relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(128, activation="relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(1, activation="sigmoid"))
        opt = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(
            loss="binary_crossentropy",
            optimizer=opt,
            metrics=["accuracy"],
        )
        return model

    def thr_to_accuracy(thr, Y_test, predictions):
        return -accuracy_score(Y_test, np.array(predictions > thr, dtype=np.int))

    def predict(self, smile):
        model = molecule.create_baseline()
        model.load_weights(os.getcwd() + "/src/model130_3.hdf5")
        fp = np.zeros((0,), dtype=int)
        fp_vec = fingerprint_features(smile)
        DataStructs.ConvertToNumpyArray(fp_vec, fp)
        result = model.predict(expand_dims(fp, 0))
        if result[0][0] > 0.65:
            return "1"
        else:
            return "0"


class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy data types"""

    def default(self, obj):
        if isinstance(
            obj,
            (
                np.int_,
                np.intc,
                np.intp,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):

            return int(obj)

        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)

        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {"real": obj.real, "imag": obj.imag}

        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()

        elif isinstance(obj, np.bool_):
            return bool(obj)

        elif isinstance(obj, np.void):
            return None

        return json.JSONEncoder.default(self, obj)


def report_metrics(scores_json, report_file):
    with open(report_file, "w") as fd:
        json.dump(scores_json, fd, indent=4, cls=NumpyEncoder)
