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
    x = tf.clip_by_value(x, 0, 1)
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