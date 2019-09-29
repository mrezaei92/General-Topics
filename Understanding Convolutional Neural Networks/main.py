from keras import applications
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt


def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x



# build the VGG16 network
model = applications.VGG16(include_top=False,
                           weights='imagenet')

# get the symbolic outputs of each "key" layer (we gave them unique names).
layer_dict = dict([(layer.name, layer) for layer in model.layers])


layer_name = 'block5_conv2'
filter_index = 28  # can be any integer from 0 to 511, as there are 512 filters in that layer

# build a loss function that maximizes the activation
# of the nth filter of the layer considered

input_img=layer_dict['block1_conv1'].input  # this is the input tensor to the network
layer_output = layer_dict[layer_name].output # this is the output of the layer

loss = K.mean(layer_output[:, :, :, filter_index])
grads = K.gradients(loss, input_img)[0]

#normalization of the gradient
grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
iterate = K.function([input_img], [loss, grads])

lr=0.01 # this is similar to the learning rate in gradient descend
input_img_data = np.random.random((1, 128, 128,3)) * 20 + 128.
# run gradient ascent for 500 steps
for i in range(500):
    loss_value, grads_value = iterate([input_img_data])
    input_img_data += grads_value * lr
    
    

## visualize the result

img = input_img_data[0]
img = deprocess_image(img)
plt.imshow(img)

