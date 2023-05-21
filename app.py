import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(__file__))

model_path = "models/generator.h5"
model = tf.keras.models.load_model(model_path)

n = 16
latent_size = 100

while True:
    input_label = int(input("Generate number?: ").strip())
    labels = tf.constant([input_label]*n)
    noise = tf.random.normal([n, latent_size])

    generated_images = model([labels, noise])

    plt.figure(figsize=(4, 4))
    plt.title(f"{input_label}s images")
    for i in range(n):
        plt.subplot(4, 4, i+1)
        plt.imshow(generated_images[i, :, :, 0], cmap="gray")
        plt.axis("off")
    plt.show()