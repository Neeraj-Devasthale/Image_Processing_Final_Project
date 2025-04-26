import tensorflow.keras as keras

def build_model(input_shape, num_classes):
    """Generates CNN model with dropout

    :param input_shape (tuple): Shape of input set
    :param num_classes (int): Number of classes for the classification task
    :return model: CNN model
    """
    inputs = keras.layers.Input(shape=input_shape)

    
    x = keras.layers.Conv2D(32, (5, 5), strides=(1, 1), padding='same', activation='relu')(inputs)
    
    
    x = keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Dropout(0.3)(x)  
    
    
    x = keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Dropout(0.3)(x)  
    
    
    x = keras.layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Dropout(0.3)(x)  
    
    
    x = keras.layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Dropout(0.3)(x)  

    
    x = keras.layers.GlobalAveragePooling2D()(x)

    
    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.Dropout(0.3)(x) 
    x = keras.layers.Dense(64, activation='relu')(x)
    x = keras.layers.Dropout(0.3)(x) 
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)

    model = keras.models.Model(inputs=inputs, outputs=outputs)
    
    return model