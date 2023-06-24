# functions to show an image
import matplotlib.pyplot as plt
import numpy as np


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def interval_mapping(image, from_min, from_max, to_min, to_max):
    from_range = from_max - from_min
    to_range = to_max - to_min
    scaled = np.array((image - from_min) / float(from_range), dtype=float)
    return to_min + (scaled * to_range)


def plot_misclassified(infer_ds, misclassified):
    samples = [infer_ds[idx][0] for idx in misclassified]
    labels = [infer_ds[idx][1] for idx in misclassified]
    f, axarr = plt.subplots(5,2, figsize=(8, 12))
    for num in range(1, 11):
        f.add_subplot(5, 2, num)
        idx = num - 1
        img = samples[idx].numpy()
        img = interval_mapping(img, np.min(img), np.max(img), 0, 255)
        plt.imshow(np.transpose(img, (1, 2, 0)).astype(np.uint8))
        plt.xlabel(infer_ds.classes[labels[idx]], fontsize=15)

    f.tight_layout()
    plt.savefig("misclassified_images.png")
    plt.show()