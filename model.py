from __future__ import absolute_import, \
    division, print_function, unicode_literals

import tensorflow as tf
import tensorflow.keras.layers as layers
import numpy as np
import scipy.io


x_data = scipy.io.loadmat('./data/x.mat')['x'].astype(np.float32)
y_data = scipy.io.loadmat('./data/y.mat')['y'].astype(np.float32)
data_count = len(y_data)

train_count = data_count // 4 * 3
test_count = data_count - train_count

train_ds = tf.data.Dataset.from_tensor_slices(
    (x_data[:train_count], y_data[:train_count])
).shuffle(train_count).batch(5120)
test_ds = tf.data.Dataset.from_tensor_slices(
    (x_data[train_count:], y_data[train_count:])
).shuffle(test_count).batch(5120)


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


model = TradFC()

loss_object = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.0003,
    beta_1=0.5,
    beta_2=0.999
)

train_loss = tf.keras.metrics.Mean(name='train_loss')
test_loss = tf.keras.metrics.Mean(name='test_loss')


@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        pred = model(x, training=True)
        loss = loss_object(y, pred)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)


@tf.function
def test_step(images, labels):
    pred = model(images, training=False)
    t_loss = loss_object(labels, pred)

    test_loss(t_loss)


EPOCHS = 20

for epoch in range(EPOCHS):
    for i, (x, y) in enumerate(train_ds):
        train_step(x, y)

    for x, y in test_ds:
        test_step(x, y)

    template = 'Epoch {}, Loss: {}, Test Loss: {}'
    print(template.format(epoch + 1, train_loss.result(), test_loss.result()))

    model.save_weights('model/model.ckpt')

    train_loss.reset_states()
    test_loss.reset_states()
