import numpy as np
import netket.graph as gr
import netket.hilbert as hs
import os
import wget
import gzip

def load_training(num_images):

    has_images = os.path.isfile('train-images-idx3-ubyte.gz')
    has_labels = os.path.isfile('train-labels-idx1-ubyte.gz')

    if (not has_images) or (not has_labels):
        print("MNIST training data incomplete or missing.")
        answer = input("Would you like to download the files (y/n)? ")
        print(answer)
        if( answer != 'y' and answer != 'Y' ):
            exit()

        url = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
        wget.download(url, 'train-images-idx3-ubyte.gz')
        url = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'
        wget.download(url, 'train-labels-idx1-ubyte.gz')


    image_size = 28

    with gzip.open("train-images-idx3-ubyte.gz", "r") as f:
        # Skip the first 16 bytes
        f.read(16)

        # Then read as many bytes as we need for num_images
        buf = f.read(image_size * image_size * num_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        imgs = data.reshape(num_images, image_size*image_size)

    with gzip.open("train-labels-idx1-ubyte.gz", "r") as f:
        # Skip the first 8 bytes
        f.read(8)

        # Then read as many bytes as we need for num_images labels
        buf = f.read(num_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        targets = data.reshape(num_images, 1)


    # 256 inputs
    g = gr.Hypercube(length=256, n_dim=1)
    hi = hs.Qubit(graph=g)

    training_samples = []
    training_targets = []
    for i in range(num_images):
        training_samples.append(imgs[i].tolist())
        training_targets.append([targets[i]])

    return hi, imgs, targets
