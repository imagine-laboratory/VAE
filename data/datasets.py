from torch.utils.data import Dataset, DataLoader

import glob
import cv2
import numpy as np
import os
import random

class PineappleDataset(Dataset):
    """
    A custom PyTorch Dataset for loading pineapple images from a directory structure,
    with optional training, validation, and test splits, as well as image augmentation.

    Args:
        test_txt (str, optional): Path to a text file containing test image base names (without extension).
        path (str): Root directory containing the image files.
        train (bool): Whether to load training data. Mutually exclusive with 'val'.
        val (bool): Whether to load validation data. Mutually exclusive with 'train'.
        train_ratio (float): Proportion of the remaining (non-test) images used for training.
        val_ratio (float): Proportion of the remaining (non-test) images used for validation.
        resize_img (int): Desired width and height for resized images.
        augment (bool): Whether to apply data augmentation during training.
        augment_ratio (int): Ratio to augment the dataset for training.
    """

    def __init__(self, test_txt="/home/rtxmsi1/Downloads/VAE_training-master (2)/test_ids_fold_1.txt", path="/home/rtxmsi1/Downloads/VAE_training-master (2)/FULL_VERTICAL_PINEAPPLE/FULL_UNIFIED",
                 train=True, val=False, train_ratio=0.8, val_ratio=0.2,
                 resize_img=256, augment=False, augment_ratio=2):
        assert not (train and val), "Only one of 'train' or 'val' should be True"

        self.path = path
        self.resize_shape = (resize_img, resize_img)
        self.train = train
        self.val = val
        self.augment = augment and train  # Only augment if training

        # Get all image file paths sorted alphabetically
        all_images = sorted(glob.glob(os.path.join(path, "*")))

        # Read test image identifiers from the test_txt file if provided
        if test_txt is not None and os.path.isfile(test_txt):
            with open(test_txt, "r") as f:
                test_ids = set(line.strip() for line in f)
        else:
            test_ids = set()

        # Map base filenames to full paths
        image_dict = {os.path.splitext(os.path.basename(img))[0]: img for img in all_images}

        # Extract test images by matching base names from test_ids
        test_images = [image_dict[id_] for id_ in test_ids if id_ in image_dict]

        # Remaining images are candidates for training/validation
        remaining_images = [img for key, img in image_dict.items() if key not in test_ids]

        # Shuffle to ensure reproducibility
        random.seed(42)
        random.shuffle(remaining_images)

        # Compute split indices
        train_end = int(train_ratio * len(remaining_images))
        val_end = train_end + int(val_ratio * len(remaining_images))

        # Select images based on split
        if train:
            self.images = remaining_images[:train_end]
        elif val:
            self.images = remaining_images[train_end:val_end]
        else:
            self.images = test_images

        # Optionally duplicate images for augmentation
        if self.augment and train:
            self.images = self.images * augment_ratio

    def __len__(self):
        """
        Returns the number of images in the current split.
        """
        return len(self.images)

    def transform_image(self, image_path):
        """
        Loads and preprocesses an image.

        Args:
            image_path (str): Path to the image file.

        Returns:
            np.ndarray: Transformed image tensor in CHW format, normalized to [0, 1].
        """
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # BGR format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)    # Convert to RGB
        image = cv2.resize(image, self.resize_shape)      # Resize image

        # Apply augmentations with probability
        if self.augment:
            if random.random() < 0.3:
                image = cv2.flip(image, 1)  # Horizontal flip

            if random.random() < 0.2:
                beta = random.uniform(-10, 10)  # Brightness shift
                image = cv2.convertScaleAbs(image, alpha=1.0, beta=beta)

            if random.random() < 0.5:
                angle = random.uniform(-10, 10)
                center = (self.resize_shape[1] // 2, self.resize_shape[0] // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1)
                image = cv2.warpAffine(image, M, self.resize_shape, borderMode=cv2.BORDER_REFLECT_101)

        # Normalize to [0, 1] and convert to CHW (Channels x Height x Width)
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        return image

    def __getitem__(self, idx):
        """
        Retrieves and transforms the image at the given index.

        Args:
            idx (int): Index of the image to retrieve.

        Returns:
            dict: Dictionary containing:
                - 'image': the preprocessed image tensor
                - 'idx': the index of the image
        """
        image = self.transform_image(self.images[idx])
        return {'image': image, 'idx': idx}