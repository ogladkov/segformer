import matplotlib.pyplot as plt


def show(nrows, ncols, ds, img_idx, figsize=(18, 10)):

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    for r in range(nrows):
        for c in range(ncols):
            ax = axes[r][c]
            ax.imshow(ds[img_idx])
            ax.axis('off')

    plt.show()
