import os
from collections import OrderedDict
import torch.utils.data as data
from . import utils


class DeepDrive(data.Dataset):
    """CamVid dataset loader where the dataset is arranged as in
    https://github.com/alexgkendall/SegNet-Tutorial/tree/master/CamVid.


    Keyword arguments:
    - root_dir (``string``): Root directory path.
    - mode (``string``): The type of dataset: 'train' for training set, 'val'
    for validation set, and 'test' for test set.
    - transform (``callable``, optional): A function/transform that  takes in
    an PIL image and returns a transformed version. Default: None.
    - label_transform (``callable``, optional): A function/transform that takes
    in the target and transforms it. Default: None.
    - loader (``callable``, optional): A function to load an image given its
    path. By default ``default_loader`` is used.

    """
    # Training dataset root folders
    train_folder = 'images/train'
    train_lbl_folder = 'labels/train'

    # Validation dataset root folders
    val_folder = 'images/val'
    val_lbl_folder = 'labels/val'

    # Test dataset root folders
    test_folder = 'images/val'
    test_lbl_folder = 'labels/val'

    # Images extension
 #   img_extension = '.png'
    img_extension = '.jpg'
    # Default encoding for pixel value, class name, and class color
    color_encoding = OrderedDict([
        (  'unlabeled' , (  0,  0,  0) ),
        (  'dynamic' , (111, 74,  0) ),
        (  'ego vehicle' , (  0,  0,  0) ),
        (  'ground' , ( 81,  0, 81) ),
        (  'static' , (  0,  0,  0) ),
        (  'parking' , (250,170,160) ),
        (  'rail track' , (230,150,140) ),
        (  'road' , (128, 64,128) ),
        (  'sidewalk' , (244, 35,232) ),
        (  'bridge' , (150,100,100) ),
        (  'building' , ( 70, 70, 70) ),
        (  'fence' , (190,153,153) ),
        (  'garage' , (180,100,180) ),
        (  'guard rail' , (180,165,180) ),
        (  'tunnel' , (150,120, 90) ),
        (  'wall' , (102,102,156) ),
        (  'banner' , (250,170,100) ),
        (  'billboard' , (220,220,250) ),
        (  'lane divider' , (255, 165, 0) ),
        (  'parking sign' , (220, 20, 60) ),
        (  'pole' , (153,153,153) ),
        (  'polegroup' , (153,153,153) ),
        (  'street light' , (220,220,100) ),
        (  'traffic cone' , (255, 70,  0) ),
        (  'traffic device' , (220,220,220) ),
        (  'traffic light' , (250,170, 30) ),
        (  'traffic sign' , (220,220,  0) ),
        (  'traffic sign frame' , (250,170,250) ),
        (  'terrain' , (152,251,152) ),
        (  'vegetation' , (107,142, 35) ),
        (  'sky' , ( 70,130,180) ),
        (  'person' , (220, 20, 60) ),
        (  'rider' , (255,  0,  0) ),
        (  'bicycle' , (119, 11, 32) ),
        (  'bus' , (  0, 60,100) ),
        (  'car' , (  0,  0,142) ),
        (  'caravan' , (  0,  0, 90) ),
        (  'motorcycle' , (  0,  0,230) ),
        (  'trailer' , (  0,  0,110) ),
        (  'train' , (  0, 80,100) ),
        (  'truck' , (  0,  0, 70) ),
    ])

    def __init__(self,
                 root_dir,
                 mode='train',
                 transform=None,
                 label_transform=None,
                 loader=utils.pil_loader):
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        self.label_transform = label_transform
        self.loader = loader

        if self.mode.lower() == 'train':
            # Get the training data and labels filepaths
            self.train_data = utils.get_files(
                os.path.join(root_dir, self.train_folder)#,
            #    extension_filter=self.img_extension)
            )
            self.train_labels = utils.get_files(
                os.path.join(root_dir, self.train_lbl_folder)#,
            #    extension_filter=self.img_extension)
            )
        elif self.mode.lower() == 'val':
            # Get the validation data and labels filepaths
            self.val_data = utils.get_files(
                os.path.join(root_dir, self.val_folder)#,
            #    extension_filter=self.img_extension)
            )
            self.val_labels = utils.get_files(
                os.path.join(root_dir, self.val_lbl_folder)#,
            #    extension_filter='.png')
            )
        elif self.mode.lower() == 'test':
            # Get the test data and labels filepaths
            self.test_data = utils.get_files(
                os.path.join(root_dir, self.test_folder)#,
            #    extension_filter=self.img_extension)
            )
            self.test_labels = utils.get_files(
                os.path.join(root_dir, self.test_lbl_folder)#,
            #    extension_filter=self.img_extension)
            )
        else:
            raise RuntimeError("Unexpected dataset mode. "
                               "Supported modes are: train, val and test")

    def __getitem__(self, index):
        """
        Args:
        - index (``int``): index of the item in the dataset

        Returns:
        A tuple of ``PIL.Image`` (image, label) where label is the ground-truth
        of the image.

        """
        if self.mode.lower() == 'train':
            data_path, label_path = self.train_data[index], self.train_labels[
            index]
        elif self.mode.lower() == 'val':
            data_path, label_path = self.val_data[index], self.val_labels[
            index]
        elif self.mode.lower() == 'test':
            data_path, label_path = self.test_data[index], self.test_labels[
            index]
        else:
            raise RuntimeError("Unexpected dataset mode. "
                               "Supported modes are: train, val and test")

        img, label = self.loader(data_path, label_path)

        if self.transform is not None:
            img = self.transform(img)

        if self.label_transform is not None:
            label = self.label_transform(label)

        return img, label


    def __len__(self):
        """Returns the length of the dataset."""
        if self.mode.lower() == 'train':
            return len(self.train_data)
        elif self.mode.lower() == 'val':
            return len(self.val_data)
        elif self.mode.lower() == 'test':
            return len(self.test_data)
        else:
            raise RuntimeError("Unexpected dataset mode. "
                               "Supported modes are: train, val and test")
