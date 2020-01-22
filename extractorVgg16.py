from keras.preprocessing import image
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model, load_model
from keras.layers import Input
import numpy as np

class Extractor():
    def __init__(self, weights=None):
        """Either load pretrained from imagenet, or load our saved
        weights from our own training."""

        self.weights = weights  
        base_model = VGG16(
            weights='imagenet',
            include_top=True
        )

        # We'll extract features at the final pool layer.
        self.model = Model(
            inputs=base_model.input,
            outputs=base_model.get_layer('fc2').output
        )

    def extract(self, image_path):
        img = image.load_img(image_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        # Get the prediction.
        features = self.model.predict(x)
        features = features[0]
        return features