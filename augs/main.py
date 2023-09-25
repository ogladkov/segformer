from base_augs import get_transforms
from dataset import SimpleDataset
from show import show


def main():

    ds_path = '../disaster_ds_raw/images/train/'
    transforms = get_transforms('train')
    ds = SimpleDataset(ds_path, transforms=transforms)
    show(nrows=3, ncols=4, ds=ds, img_idx=3)


if __name__ == '__main__':

    main()