def rotate_random(image):
    # random horizontal flipping.
    image = tf.image.random_flip_left_right(image)
	image = tf.image.random_flip_up_down(image)
    return image


def color_jitter(x, s=0.5):
    x = tf.image.random_brightness(x, max_delta=0.8*s)
    x = tf.image.random_contrast(x, lower=1-0.8*s, upper=1+0.8*s)
    x = tf.image.random_saturation(x, lower=1-0.8*s, upper=1+0.8*s)
    x = tf.image.random_hue(x, max_delta=0.2*s)
    x = tf.clip_by_value(x, 0, 255)
    return x
	
    # Kind of similar to color_jitter
def random_color_saturation(image):
	image = tf.image.random_saturation(image)
	return image
	
    
    # Randomly sets the RGB value of an image to 0
def get_random_pixel_drop(drop_num, l_bound = 0, u_bound = 255):
    def drop(input_img):    
        if len(input_img) == 0:
            print('invalid_image')
            
        img_h, img_w, img_c = input_img.shape
        
        for drop in range(drop_num):
            rand_x = random.randint(0,img_h-1)
            rand_y = random.randint(0,img_w-1)
            
            left = np.random.randint(0, img_w)
            top = np.random.randint(0, img_h)
            c = random.randint(l_bound,u_bound)
            input_img[rand_x, rand_y, :]= c
        return input_img
    return drop


    # Not my code -- 
def gaussian_blur(image, kernel_size=23, padding='SAME'):
    sigma = tf.random.uniform((1,))* 1.9 + 0.1

    radius = tf.cast(kernel_size / 2, tf.int32)
    kernel_size = radius * 2 + 1
    x = tf.cast(tf.range(-radius, radius + 1), tf.float32)
    blur_filter = tf.exp(
        -tf.pow(x, 2.0) / (2.0 * tf.pow(tf.cast(sigma, tf.float32), 2.0)))
    blur_filter /= tf.reduce_sum(blur_filter)
    # One vertical and one horizontal filter.
    blur_v = tf.reshape(blur_filter, [kernel_size, 1, 1, 1])
    blur_h = tf.reshape(blur_filter, [1, kernel_size, 1, 1])
    num_channels = tf.shape(image)[-1]
    blur_h = tf.tile(blur_h, [1, 1, num_channels, 1])
    blur_v = tf.tile(blur_v, [1, 1, num_channels, 1])
    expand_batch_dim = image.shape.ndims == 3
    if expand_batch_dim:
        image = tf.expand_dims(image, axis=0)
    blurred = tf.nn.depthwise_conv2d(
        image, blur_h, strides=[1, 1, 1, 1], padding=padding)
    blurred = tf.nn.depthwise_conv2d(
        blurred, blur_v, strides=[1, 1, 1, 1], padding=padding)
    if expand_batch_dim:
        blurred = tf.squeeze(blurred, axis=0)
    return blurred
