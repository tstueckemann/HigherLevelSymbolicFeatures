import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Flatten, Dense

class ResidualBlock(tf.keras.layers.Layer):
    """ The Residual block of ResNet models. """
    def __init__(self, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = Conv2D(num_channels, kernel_size=3, padding='same', strides=strides)
        self.conv2 = Conv2D(num_channels, kernel_size=3, padding='same')
        self.bn1 = BatchNormalization()
        self.bn2 = BatchNormalization()
        self.conv3 = Conv2D(num_channels, kernel_size=1, strides=strides) if use_1x1conv else None

    def call(self, X):
        # First convolution block
        Y = self.conv1(X)
        Y = self.bn1(Y)
        Y = ReLU()(Y)

        # Second convolution block
        Y = self.conv2(Y)
        Y = self.bn2(Y)

        # Residual connection
        if self.conv3:
            X = self.conv3(X)

        Y += X
        return ReLU()(Y)

class ResNet(tf.keras.Model):
    """ The ResNet model. """
    def __init__(self, num_resblocks, num_channels, num_classes, use_1x1conv=True, strides=1):
        super().__init__()
        self.res_blocks = [ResidualBlock(num_channels, use_1x1conv, strides) for _ in range(num_resblocks)]
        self.conv = Conv2D(num_channels, kernel_size=1, padding='same')
        self.flatten = Flatten()
        self.fc = Dense(num_classes, activation='softmax')

    def call(self, X):
        for blk in self.res_blocks:
            X = blk(X)
        X = self.conv(X)
        X = self.flatten(X)
        X = self.fc(X)
        return X
