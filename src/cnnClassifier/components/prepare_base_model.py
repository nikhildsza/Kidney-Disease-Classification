import os
import tensorflow as tf
from zipfile import ZipFile
import urllib.request as request
from pathlib import Path
from cnnClassifier.entity.config_entity import PrepareBaseModelConfig

class PrepareBaseModel:
    def __init__(self,config:PrepareBaseModelConfig):
        self.config = config

    def get_base_model(self):
        self.model = tf.keras.applications.vgg16.VGG16(
            input_shape = self.config.params_image_size,
            weights = self.config.params_weights,
            include_top = self.config.params_include_top
        )

        self.save_model(path=self.config.base_model_path,model=self.model)
    
    @staticmethod
    def save_model(path:Path, model:tf.keras.Model):
        model.save(path)

    @staticmethod
    def prepare_base_model(model,freeze_all,freeze_till,classes,learning_rate):
        if freeze_all:
            for layer in model.layers:
                layer.trainable = False
        elif(freeze_till is not None) and (freeze_till>0):
            for layer in model.layers[:-freeze_till]:
                layer.trainable = False

        flatten_layer = tf.keras.layers.Flatten()
        flatten_out = flatten_layer(model.output)
        
        fc1 = tf.keras.layers.Dense(
            units=128,
            activation='relu'
        )(flatten_out)
        fc2 = tf.keras.layers.Dense(
            units=64,
            activation='relu'
        )(fc1)
        fc3 = tf.keras.layers.Dense(
            units=32,
            activation='relu'
        )(fc2)
        predictions = tf.keras.layers.Dense(
            units=classes,
            activation='sigmoid'
        )(fc3)

        final_model = tf.keras.models.Model(
            inputs = model.input,
            outputs = predictions
        )

        final_model.compile(
            optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate),
            loss = 'binary_crossentropy',
            metrics = ['accuracy']
        )
        final_model.summary()
        return final_model
    
    def get_updated_model(self):
        self.full_model = self.prepare_base_model(
            model = self.model,
            freeze_all = True,
            freeze_till = None,
            classes = self.config.params_classes,
            learning_rate = self.config.params_learning_rate
        )
        self.save_model(path=self.config.updated_base_model_path,model=self.full_model)