import glob
import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.image as mpimg
import pandas as pd
import cv2


class FacialKeypointsDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.key_pts_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.key_pts_frame)

    def __getitem__(self, idx):
        image_name = os.path.join(self.root_dir,
                                self.key_pts_frame.iloc[idx, 0])
        
        image = mpimg.imread(image_name)
        
        # if image has an alpha color channel, get rid of it
        if(image.shape[2] == 4):
            image = image[:,:,0:3]
        
        key_pts = self.key_pts_frame.iloc[idx, 1:].values
        key_pts = key_pts.astype('float').reshape(-1, 2)
        sample = {'image': image, 'keypoints': key_pts}

        if self.transform:
            sample = self.transform(sample)

        return sample
    

    
# tranforms

class Normalize(object):
    """Convert a color image to grayscale and normalize the color range to [0,1]."""        

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
        
        image_copy = np.copy(image)
        key_pts_copy = np.copy(key_pts)

        # convert image to grayscale
        image_copy = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # scale color range from [0, 255] to [0, 1]
        # (newer matplotlib versions already return images as floats in [0, 1])
        if image_copy.max() > 1.0:
            image_copy = image_copy/255.0
            
        
        # scale keypoints to be centered around 0 with a range of [-1, 1]
        # mean = 100, sqrt = 50, so, pts should be (pts - 100)/50
        key_pts_copy = (key_pts_copy - 100)/50.0


        return {'image': image_copy, 'keypoints': key_pts_copy}


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = cv2.resize(image, (new_w, new_h))
        
        # scale the pts, too
        key_pts = key_pts * [new_w / w, new_h / h]

        return {'image': img, 'keypoints': key_pts}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        key_pts = key_pts - [left, top]

        return {'image': image, 'keypoints': key_pts}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
         
        # if image has no grayscale color channel, add one
        if(len(image.shape) == 2):
            # add that third color dim
            image = image.reshape(image.shape[0], image.shape[1], 1)
            
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        
        return {'image': torch.from_numpy(image),
                'keypoints': torch.from_numpy(key_pts)}

# data augmentation transforms (training only)

class RandomRotate(object):
    """Rotate the image and keypoints by a random angle in [-max_angle, max_angle] degrees."""

    def __init__(self, max_angle=15):
        self.max_angle = max_angle

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

        angle = np.random.uniform(-self.max_angle, self.max_angle)
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)

        image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        # apply the same affine transform to the (x, y) keypoints
        key_pts = np.hstack([key_pts, np.ones((key_pts.shape[0], 1))]).dot(M.T)

        return {'image': image, 'keypoints': key_pts}


# index of the mirrored keypoint for each of the 68 points (iBUG 300-W layout)
FLIP_IDX = np.array(
    list(range(16, -1, -1)) +                       # jaw 0-16
    list(range(26, 21, -1)) + list(range(21, 16, -1)) +  # eyebrows 17-26
    [27, 28, 29, 30] +                              # nose bridge
    [35, 34, 33, 32, 31] +                          # nostrils
    [45, 44, 43, 42, 47, 46] +                      # right eye <- left eye
    [39, 38, 37, 36, 41, 40] +                      # left eye <- right eye
    [54, 53, 52, 51, 50, 49, 48, 59, 58, 57, 56, 55] +  # outer lips
    [64, 63, 62, 61, 60, 67, 66, 65]                # inner lips
)


class RandomHorizontalFlip(object):
    """Mirror the image horizontally with probability p, swapping left/right keypoints."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

        if np.random.rand() < self.p:
            w = image.shape[1]
            image = np.ascontiguousarray(image[:, ::-1])
            key_pts = key_pts.copy()
            key_pts[:, 0] = w - 1 - key_pts[:, 0]
            # a left eye point becomes a right eye point, etc.
            key_pts = key_pts[FLIP_IDX]

        return {'image': image, 'keypoints': key_pts}


class RandomBrightnessContrast(object):
    """Randomly change brightness and contrast. Use after Normalize (image in [0, 1])."""

    def __init__(self, brightness=0.2, contrast=0.2):
        self.brightness = brightness
        self.contrast = contrast

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

        alpha = 1.0 + np.random.uniform(-self.contrast, self.contrast)
        beta = np.random.uniform(-self.brightness, self.brightness)
        image = np.clip(alpha * (image - 0.5) + 0.5 + beta, 0.0, 1.0)

        return {'image': image, 'keypoints': key_pts}


class CenterCrop(object):
    """Crop the center of the image (deterministic, for evaluation)."""

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

        h, w = image.shape[:2]
        top = (h - self.output_size) // 2
        left = (w - self.output_size) // 2

        image = image[top: top + self.output_size,
                      left: left + self.output_size]
        key_pts = key_pts - [left, top]

        return {'image': image, 'keypoints': key_pts}
