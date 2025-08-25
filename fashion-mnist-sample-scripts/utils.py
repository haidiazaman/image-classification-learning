import matplotlib.pyplot as plt

def plot_image_grid(dataset, num_rows=3, num_cols=5):
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols*2, num_rows*2))
    axes = axes.flatten()

    for i in range(num_rows * num_cols):
        img, label = dataset[i]
        img = img.permute(1,2,0)
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(label.item())
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()