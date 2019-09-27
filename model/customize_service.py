import numpy as np
from model_service.tfserving_model_service import TfServingBaseService
import pandas as pd
import tensorflow as tf
import tensorflow.keras.layers as layers
import os


class TradFC(tf.keras.Model):

    def __init__(self):
        super(TradFC, self).__init__()

        self.bn0 = layers.BatchNormalization()

        self.fc1 = layers.Dense(
            256,
            kernel_initializer='he_normal',
            kernel_regularizer=tf.keras.regularizers.l2(0.01),
            activity_regularizer=tf.keras.regularizers.l1(0.01)
        )
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.LeakyReLU(0.2)
        self.dropout1 = layers.Dropout(0.3)

        self.fc2 = layers.Dense(
            256,
            kernel_initializer='he_normal',
            kernel_regularizer=tf.keras.regularizers.l2(0.01),
            activity_regularizer=tf.keras.regularizers.l1(0.01)
        )
        self.bn2 = layers.BatchNormalization()
        self.relu2 = layers.LeakyReLU(0.2)
        self.dropout2 = layers.Dropout(0.3)

        self.fc3 = layers.Dense(
            256,
            kernel_initializer='he_normal',
            kernel_regularizer=tf.keras.regularizers.l2(0.01),
            activity_regularizer=tf.keras.regularizers.l1(0.01)
        )
        self.bn3 = layers.BatchNormalization()
        self.relu3 = layers.LeakyReLU(0.2)
        self.dropout3 = layers.Dropout(0.3)

        self.fc4 = layers.Dense(
            256,
            kernel_initializer='he_normal',
            kernel_regularizer=tf.keras.regularizers.l2(0.01),
            activity_regularizer=tf.keras.regularizers.l1(0.01)
        )
        self.bn4 = layers.BatchNormalization()
        self.relu4 = layers.LeakyReLU(0.2)
        self.dropout4 = layers.Dropout(0.3)

        self.fc5 = layers.Dense(
            256,
            kernel_initializer='he_normal',
            kernel_regularizer=tf.keras.regularizers.l2(0.01),
            activity_regularizer=tf.keras.regularizers.l1(0.01)
        )
        self.bn5 = layers.BatchNormalization()
        self.relu5 = layers.LeakyReLU(0.2)
        self.dropout5 = layers.Dropout(0.3)

        self.fc7 = layers.Dense(1)

    def call(self, x, training=True):

        x = self.bn0(x, training=training)

        x = self.fc1(x)
        x = self.bn1(x, training=training)
        x = self.relu1(x)
        x = self.dropout1(x, training=training)
        r1 = x

        out = self.fc2(x)
        out = self.bn2(out, training=training)
        out = self.relu2(out)
        out = self.dropout2(out, training=training)

        out = self.fc3(out)
        out = self.bn3(out, training=training)
        out = out + r1
        out = self.relu3(out)
        out = self.dropout3(out, training=training)
        r2 = out

        out = self.fc4(out)
        out = self.bn4(out, training=training)
        out = self.relu4(out)
        out = self.dropout4(out, training=training)

        out = self.fc5(out)
        out = self.bn5(out, training=training)
        out = out + r2
        out = self.relu5(out)
        out = self.dropout5(out, training=training)

        out = self.fc7(out)

        return out


class my_service(TfServingBaseService):

    def __init__(self, model_name, model_path):
        self.model_name = model_name
        print(model_name)
        self.model_path = model_path
        print(model_path)
        self.model = TradFC()
        self.model.load_weights(os.path.join(model_path, 'model.ckpt'))

    def _preprocess(self, data):
        print(vars(self))
        preprocessed_data = {}
        filesDatas = []
        for k, v in data.items():
            for file_name, file_content in v.items():
                pb_data = pd.read_csv(file_content)
                input_data = np.array(
                    pb_data.get_values()[:, 0:17], dtype=np.float32)
                print(file_name, input_data.shape)
                filesDatas.append(input_data)

        raw_data = np.array(filesDatas, dtype=np.float32).reshape(-1, 17)

        d = (np.sqrt((raw_data[:, 1] - raw_data[:, 12]) ** 2
                     + (raw_data[:, 2] - raw_data[:, 13]) ** 2) / 1000)

        pl = raw_data[:, 8] \
            - (46.3 + 33.9 * np.log10(raw_data[:, 7])
                - 13.82 * np.log10(raw_data[:, 3] + raw_data[:, 9] + 1)
                + (44.9 - 6.55
                   * np.log10(raw_data[:, 15] + raw_data[:, 14] + 1) - 30)
                * np.log10(d + 1) - 15)

        hb = raw_data[:, 3] + raw_data[:, 9] - raw_data[:, 14]
        hv = hb - d * 1000 * np.tan((raw_data[:, 5]
                                     + raw_data[:, 6]) * np.pi / 180)

        nx = raw_data[:, 12] - raw_data[:, 1]
        ny = raw_data[:, 13] - raw_data[:, 2]

        a = np.zeros_like(nx)
        a[((ny == 0).astype(np.int) * (nx > 0).astype(np.int))
            .astype(np.bool)] = 90
        a[((ny == 0).astype(np.int) * (nx < 0).astype(np.int))
            .astype(np.bool)] = 270
        idxa = ny > 0
        a[idxa] = np.arctan(nx[idxa] / ny[idxa]) / np.pi * 180
        idxa = ny < 0
        a[idxa] = 180 - np.arctan(nx[idxa] / ny[idxa]) / np.pi * 180
        a = np.abs(a - raw_data[:, 4])
        idxa = a >= 360
        a[idxa] = a[idxa] - 360
        idxa = a > 180
        a[idxa] = 360 - a[idxa]

        nx = nx / 5
        ny = ny / 5

        data = np.concatenate([
            nx[:, np.newaxis],
            ny[:, np.newaxis],
            raw_data[:, 3][:, np.newaxis],
            raw_data[:, 4][:, np.newaxis],
            a[:, np.newaxis],
            raw_data[:, 5][:, np.newaxis],
            raw_data[:, 6][:, np.newaxis],
            raw_data[:, 7][:, np.newaxis],
            raw_data[:, 8][:, np.newaxis],
            raw_data[:, 9][:, np.newaxis],
            raw_data[:, 10][:, np.newaxis],
            raw_data[:, 11][:, np.newaxis],
            d[:, np.newaxis],
            hv[:, np.newaxis],
            raw_data[:, 14][:, np.newaxis],
            raw_data[:, 15][:, np.newaxis],
            raw_data[:, 16][:, np.newaxis],
            pl[:, np.newaxis]
        ], axis=1)

        preprocessed_data['myInput'] = data
        print("preprocessed_data[\'myInput\'].shape = ",
              preprocessed_data['myInput'].shape)

        return preprocessed_data

    def _postprocess(self, data):
        infer_output = {"RSRP": []}
        for output_name, results in data.items():
            print(output_name, np.array(results).shape)
            infer_output["RSRP"] = results
        return infer_output

    def _inference(self, data):
        print(data['myInput'].shape)
        output = self.model(data['myInput'], training=False)
        return {'output': output.numpy().tolist()}
