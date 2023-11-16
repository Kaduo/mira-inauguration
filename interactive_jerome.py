import matplotlib.pyplot as plt
import numpy as np
from fdog import difference_of_gaussians
from skimage.filters import threshold_otsu, threshold_mean
from skimage.morphology import dilation
import skimage as ski

from matplotlib.widgets import Button, Slider


def big_dawg(image, sigma, p, dilation_size=1):
    if dilation_size == 1:
        footprint = None
    else:
        footprint = np.ones((dilation_size, dilation_size))
    dog_response = ski.color.rgb2gray(
        difference_of_gaussians(image, sigma, high_sigma=1.7 * sigma, p=p)
    )
    thresh_otsu = dog_response > threshold_otsu(dog_response)
    edge_image_otsu = 1 - (thresh_otsu ^ dilation(thresh_otsu, footprint=footprint))
    thresh_mean = dog_response > threshold_mean(dog_response)
    edge_image_mean = 1 - (thresh_mean ^ dilation(thresh_mean, footprint=footprint))
    canny_edge_image = 1 - ski.feature.canny(ski.color.rgb2gray(image))
    # canny_edge_image = edge_image_mean
    return dog_response, thresh_otsu, thresh_mean, edge_image_otsu, edge_image_mean, canny_edge_image


init_sigma = 5
init_p = 5
init_dilation_size = 3


image = ski.io.imread("data/leccia.jpg")

dog_response, thresh_otsu, thresh_mean, edge_image_otsu, edge_image_mean , canny_edge_image= big_dawg(
    image, sigma=init_sigma, p=init_p, dilation_size=init_dilation_size
)

# Create the figure and the line that we will manipulate
fig, ax = plt.subplots(2, 3)
im_dog = ax[0, 0].imshow(dog_response, cmap="gray")
im_thresh_otsu = ax[0, 1].imshow(thresh_otsu, cmap="gray")
im_edge_otsu = ax[0, 2].imshow(edge_image_otsu, cmap="gray")
ax[1,0].imshow(canny_edge_image, cmap="gray")
im_thresh_mean = ax[1, 1].imshow(thresh_mean, cmap="gray")
im_edge_mean = ax[1, 2].imshow(edge_image_mean, cmap="gray")

# adjust the main plot to make room for the sliders
fig.subplots_adjust(left=0.25, bottom=0.25)

# Make a horizontal slider to control the frequency.
axsigma = fig.add_axes([0.25, 0.1, 0.65, 0.03])
sigma_slider = Slider(
    ax=axsigma,
    label="Low sigma",
    valmin=0.1,
    valmax=30,
    valinit=init_sigma,
)

axp = fig.add_axes([0.25, 0.065, 0.65, 0.03])
p_slider = Slider(
    ax=axp,
    label="p",
    valmin=0,
    valmax=50,
    valinit=init_p,
)

axdilate = fig.add_axes([0.25, 0.03, 0.65, 0.03])
dilate_slider = Slider(
    ax=axdilate,
    label="Dilation size",
    valmin=1,
    valmax=10,
    valstep=list(range(1, 11)),
    valinit=init_dilation_size,
)


# The function to be called anytime a slider's value changes
def update(val):
    dog_response, thresh_otsu, thresh_mean, edge_image_otsu, edge_image_mean, canny_edge_image = big_dawg(
        image, sigma=sigma_slider.val, p=p_slider.val, dilation_size=int(dilate_slider.val)
    )
    im_dog.set_data(dog_response)
    im_thresh_otsu.set_data(thresh_otsu)
    im_edge_otsu.set_data(edge_image_otsu)
    im_thresh_mean.set_data(thresh_mean)
    im_edge_mean.set_data(edge_image_mean)

    fig.canvas.draw_idle()


# register the update function with each slider
sigma_slider.on_changed(update)
p_slider.on_changed(update)
dilate_slider.on_changed(update)


plt.show()
