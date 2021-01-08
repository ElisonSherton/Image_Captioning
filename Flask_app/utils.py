# import necessary libraries
import os                                                       # To deal with filepaths and repositories
import random                                                   # For generating random numbers or randomly selecting one out of a bunch of images

import numpy as np                                              # Numerical Python for array, matrix operations 

from keras.models import load_model, Model                      # To load the pretrained model and to define a modified ResNet50 architecture for image encoding
from keras.preprocessing.text import Tokenizer                  # To load vocabulary for generating words
from keras.applications.resnet import ResNet50                  # For image encoding
from keras.applications.resnet50 import preprocess_input        # To transform image into the format expected by ResNet50 architecture (Size, Shape, Normalization etc.)
from keras.preprocessing.image import load_img, img_to_array    # To load image into memory and to convert image to numpy array
from keras.preprocessing.sequence import pad_sequences          # To generate sequences of a fixed length (pad the sequence if there's less words with enough fillers)

from pickle import load                                         # To load the tokenizer object saved on disk

class utils:
    @staticmethod
    def get_image_features(img_abs_path):
        '''
        Given the path where the image is stored,
        pass it through a ResNet50 imagenet trained model and
        return the features for the same
        '''
        model = ResNet50()                                                            # Instantiate a ResNet50 model
        model.layers.pop()                                                            # Remove the last layer of pretrained ImageNet data
        Image_Feature_Generator = Model(inputs = model.inputs, outputs = model.layers[-1].output)  # Redefine the model after removing the last layer
        img = load_img(img_abs_path, target_size = (224, 224))                        # Load the image from the path and resize to (224,224) for passing it through the ResNet pretrained model
        img = img_to_array(img)                                                       # Convert the image into a numpy array
        rows, columns, channels = img.shape                                           # Extract the dimensions of the image
        img = img.reshape((1, rows, columns, channels))                               # Redefine the image dimensions in a batch format
        img = preprocess_input(img)                                                   # Preprocess the input in order to bring it in a similar format as the imagenet preprocessing steps in original ResNet architecture
        features = Image_Feature_Generator.predict(img)                               # Generate features from the image
        return features
    
    @staticmethod
    def load_pretrained_data(model_path, tokenizer_path):
        '''
        Given the path for model and tokenizer, extracts
        the same and returns model, tokenizer and maximum 
        caption length (specified manually)
        '''
        model = load_model(model_path)
        tk = load(open(tokenizer_path, 'rb'))
        max_length = 22
        return (model, tk, max_length)
    
    @staticmethod
    def get_word_from_idx(idx, tokenizer):
        '''
        Given the tokenizer and the index of a predicted
        word, returns the word by hunting for the same in 
        the tokenizer's dictionary
        '''
        idx_to_word = {val:key for key, val in tokenizer.word_index.items()}
        if idx in idx_to_word.keys():
            return idx_to_word[idx]
        else:
            return None
    
    @staticmethod
    def generate_caption(model_path, image_path, tokenizer_path):
        '''
        Given the path for image, pretrained model, and
        tokenizer, it generates and subsequently returns a captions
        '''
        ip_seq = 'startseq'                                                           # Seeding the caption generation with startseq
        model, tk, max_length = self.load_pretrained_data(model_path, tokenizer_path) # Retrieve the model, tokenizer, max caption length from provided path
        features = self.get_image_features(image_path)                                # Extract ResNet50 features corresponding to the image

        for idx in range(max_length):
            seq = tk.texts_to_sequences([ip_seq])[0]                               # Tokenize the input using the tokenizer object loaded above
            seq = pad_sequences([seq], maxlen = max_length, padding = 'post')      # Pad the sequences to be of same length as the maximum caption length
            new_word_distribution = model.predict([features, seq], verbose = 0)    # Forward propogate the image features and caption features
            new_word_idx = np.argmax(new_word_distribution)                        # Get the index corresponding to most probable word outcome
            new_word = self.get_word_from_idx(new_word_idx, tk)                    # Extract the word from index generated above

            if new_word:
                if new_word != 'endseq':                                              
                    ip_seq = ip_seq + " " + new_word                                   # If the new word isn't None or endseq, then add it to the ip_seq for further caption generation
                else:
                    break                                                              # If word is 'endseq' then break
            else:
                break                                                              # If no word is returned then break

        op_seq = " ".join([i for i in ip_seq.split(' ') if i!='startseq' and i!= 'endseq']) # Remove the words startseq and endseq from the ouput if they're present
        return op_seq
