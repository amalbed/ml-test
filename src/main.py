import operator
import os
import pandas as pd
import scipy


from .feature_extractor import fingerprint_features


import tensorflow as tf


from tensorflow.keras import layers

from .model_manager import molecule, report_metrics

from rdkit import Chem
from tensorflow.python.ops.numpy_ops import np_config

from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold


path = os.getcwd()
classification_reports_path = os.path.join(path, "classification_reports")

df = pd.read_csv(path + "/data/dataset_single.csv", sep=",")
df = df.drop_duplicates()

spliter = StratifiedKFold(n_splits=5, random_state=123, shuffle=True)


def model_1():

    # Morgan Fingerprint (ECFPx)
    nBits = 2048
    mfp_name = [f"Bit_{i}" for i in range(nBits)]
    mfp_bits = [list(l) for l in df.smiles.apply(lambda x: fingerprint_features(x))]
    df_morgan = pd.DataFrame(mfp_bits, index=df.smiles, columns=mfp_name)
    df_morgan_ = df_morgan.copy()
    df_morgan_.reset_index(inplace=True)
    df_morgan_["target"] = df["P1"]
    del df_morgan_["smiles"]
    y = df_morgan_["target"]
    del df_morgan_["target"]
    X = df_morgan_.values
    y = y.values

    results = {}
    best_threshold = {}
    # for threshold in [1,50,80,90,100,110,120,130,140]:
    for threshold in [120, 130, 140]:
        for index, (train, test) in enumerate(spliter.split(X, y)):
            class_weight = {0: threshold, 1: 1}
            my_callbacks = [
                tf.keras.callbacks.EarlyStopping(patience=5),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=0.1,
                    patience=2,
                    verbose=1,
                    mode="min",
                    min_delta=0.0001,
                    cooldown=0,
                    min_lr=1e-6,
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    f"model_1/model{threshold}_{index}.hdf5",
                    monitor="val_loss",
                    verbose=1,
                    save_best_only=True,
                    mode="min",
                    save_freq="epoch",
                ),
            ]

            # train
            model = molecule.create_baseline()
            model.fit(
                x=X[train],
                y=y[train],
                batch_size=256,
                epochs=50,
                verbose="auto",
                callbacks=my_callbacks,
                validation_data=(X[test], y[test]),
                shuffle=True,
                workers=1,
                use_multiprocessing=True,
                class_weight=class_weight,
            )
            model.load_weights(f"model_1/model{threshold}_{index}.hdf5")

            # predict
            result = model.predict(X[test])

            # evaluate
            best_thr = scipy.optimize.fmin(
                molecule.thr_to_accuracy, args=(y[test], result), x0=0.5
            )
            results[f"model_1/model{threshold}_{index}.hdf5"] = accuracy_score(
                y[test], result > best_thr
            )
            best_threshold[f"model_1/model{threshold}_{index}.hdf5"] = best_thr

    model.load_weights(max(results.items(), key=operator.itemgetter(1))[0])
    print(
        "Best model : ",
        max(results.items(), key=operator.itemgetter(1))[0],
        " , and its accuracy score : ",
        results.get(max(results.items(), key=operator.itemgetter(1))[0]),
    )

    report = classification_report(
        y[test],
        result
        > best_threshold.get(max(results.items(), key=operator.itemgetter(1))[0]),
        output_dict=True,
    )
    report_metrics(
        report, os.path.join(classification_reports_path, "classification_model_1.json")
    )


def array_of_smiles(element):
    x = Chem.MolFromSmiles(element.numpy()[0])._repr_png_()
    return tf.io.decode_image(x, dtype=tf.dtypes.uint8)


class transformer_layer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(self)

    def call(self, inputs):
        return tf.map_fn(
            fn=lambda x: array_of_smiles(x),
            elems=inputs.reshape(-1, 1),
            dtype=tf.dtypes.uint8,
        )


class model_smile(tf.keras.Model):
    def __init__(self):
        super().__init__(self)
        self.transformer_layer = transformer_layer()
        self.l1 = layers.Conv2D(
            32, (3, 3), activation="relu", input_shape=(150, 450, 3)
        )
        self.l2 = layers.MaxPooling2D((2, 2))
        self.l3 = layers.Conv2D(64, (3, 3), activation="relu")
        self.l4 = layers.MaxPooling2D((2, 2))
        self.l5 = layers.Conv2D(64, (3, 3), activation="relu")
        self.l6 = layers.Flatten()
        self.l7 = layers.Dense(64, activation="relu")
        self.l8 = layers.Dense(1)

    def call(self, inputs, training=False):
        x = self.transformer_layer(inputs)
        x = x / 255
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)


def model_2():

    tf.compat.v1.enable_eager_execution()
    np_config.enable_numpy_behavior()

    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    X = df.smiles.values
    y = df.P1.values

    results = {}
    best_threshold = {}
    # for threshold in [1,50,80,90,100,110,120,130,140]:
    for threshold in [130, 140]:
        for index, (train, test) in enumerate(spliter.split(X, y)):
            class_weight = {0: threshold, 1: 1}
            my_callbacks = [
                tf.keras.callbacks.EarlyStopping(patience=5),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=0.1,
                    patience=2,
                    verbose=1,
                    mode="min",
                    min_delta=0.0001,
                    cooldown=0,
                    min_lr=1e-6,
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    f"model2mini/model{threshold}_{index}.hdf5",
                    monitor="val_loss",
                    verbose=1,
                    save_weights_only=True,
                    save_best_only=True,
                    mode="min",
                    save_freq="epoch",
                ),
            ]
            # train
            model = model_smile()
            model.compile(
                loss="binary_crossentropy",
                optimizer=opt,
                metrics=["accuracy"],
                run_eagerly=True,
            )
            model.fit(
                x=X[train],
                y=y[train],
                batch_size=256,
                epochs=5,
                verbose="auto",
                callbacks=my_callbacks,
                validation_data=(X[test], y[test]),
                shuffle=True,
                workers=1,
                use_multiprocessing=True,
                class_weight=class_weight,
            )
            model.load_weights(f"model2mini/model{threshold}_{index}.hdf5")
            # predict
            result = model.predict(X[test])
            # evaluate
            best_thr = scipy.optimize.fmin(
                molecule.thr_to_accuracy, args=(y[test], result), x0=0.5
            )
            results[f"model2mini/model{threshold}_{index}.hdf5"] = accuracy_score(
                y[test], result > best_thr
            )
            best_threshold[f"model2mini/model{threshold}_{index}.hdf5"] = best_thr

    model.load_weights(max(results.items(), key=operator.itemgetter(1))[0])
    print(
        "Best model : ",
        max(results.items(), key=operator.itemgetter(1))[0],
        " , and its accuracy score : ",
        results.get(max(results.items(), key=operator.itemgetter(1))[0]),
    )

    report = classification_report(
        y[test],
        result
        > best_threshold.get(max(results.items(), key=operator.itemgetter(1))[0]),
        output_dict=True,
    )
    report_metrics(
        report, os.path.join(classification_reports_path, "classification_model_2.json")
    )


if __name__ == "__main__":

    model_1()
