from numpy.random import randn
from tensorflow.keras.models import load_model
from matplotlib import pyplot

### generate points in latent space
def generateLatentPoints(latent_dim, n_samples):
    x_input = randn(latent_dim * n_samples)
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input

### create and save a plot 5 * 5 of generated images in reversed grayscale
def savePlot(examples, n):
    for i in range(n * n):
        pyplot.subplot(n, n, 1 +i)
        pyplot.axis('off')
        pyplot.imshow(examples[i, :, :, 0], cmap='gray_r')
        pyplot.show()
        pyplot.savefig('5X5_generated_images.png')

### Load our saved Model
model = load_model('generator_model_0100.keras')
### generate latent points
latent_points = generateLatentPoints(100, 25)
### generate images
images = model.predict(latent_points)
### save images
savePlot(examples, 5)
