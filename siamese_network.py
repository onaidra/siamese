from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Conv2D,Dense,Dropout,GlobalAveragePooling2D,MaxPooling2D,Flatten
from tensorflow.keras.applications.resnet50 import ResNet50
def build_siamese_model(inputShape,embeddingDim=48):

	# specify the inputs for the feature extractor network
    inputs = Input(inputShape)
	# define the first set of CONV => RELU => POOL => DROPOUT layers
    x = ResNet50(include_top=False, weights='imagenet', input_tensor=inputs, pooling=max)
    #x = Conv2D(64, (2, 2), padding="same", activation="relu")(inputs)
	#x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.3)(x)
	# second set of CONV => RELU => POOL => DROPOUT layers
	#x = Conv2D(64, (2, 2), padding="same", activation="relu")(x)
    x = ResNet50(include_top=False, weights='imagenet', input_tensor=x, pooling=max)
	#x = MaxPooling2D(pool_size=2)(x)
    x = Dropout(0.3)(x)

	# prepare the final outputs
	#pooledOutput = GlobalAveragePooling2D()(x)
	#outputs = Dense(embeddingDim)(pooledOutput)
	# build the model
	#model = Model(inputs, outputs)
	# return the model to the calling function
    return x
"""
    I1 = Input(inputShape)
    print("##########################################################################################")
    model = ResNet50(include_top=False, weights='imagenet', input_tensor=I1, pooling=max)
    model.layers.pop()
    model.outputs = [model.layers[-1].output]
    model.layers[-1]._outbound_nodes = []
    print("------------------------------------------------------------------------------------------")
    for layer in model.layers:
        layer._name = layer.name + str(suffix)
        layer._trainable = False

    flatten_name = 'flatten' + str(suffix)

    x = model.output

    x = Flatten(name=flatten_name)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(dropout_rate)(x)

    return x, model.input    
    """