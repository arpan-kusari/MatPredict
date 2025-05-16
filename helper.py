import numpy as np
import matplotlib.pyplot as plt
def show_array_as_image(arr, title=None):
   
    plt.figure()
    if title:
        plt.title(title)
    img_arr = np.transpose(arr,(1,2,0))
    plt.imshow(img_arr)
    plt.axis('off')   
    plt.show()
