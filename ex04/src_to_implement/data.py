from torch.utils.data import Dataset
import torchvision as tv
from skimage.io import imread
from skimage.color import gray2rgb


train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


class ChallengeDataset(Dataset):
    def __init__(self, data, mode):
        # data = pandas.dataframe, a container structure that stores the information found in the file "data.csv".
        # val = a flag of type String which can be either "val" or "train"

        self.data = data  # information inside "data.csv". It is read somewhere else and result comes as parameter
        self.mode = mode
        self.image_names = data['filename'].values
        self.labels = data[['crack', 'inactive']].values

        # Depending on the mode, we do data augmentation
        if self.mode == "train":
            self._transform = tv.transforms.Compose([
                tv.transforms.ToPILImage(),
                tv.transforms.RandomVerticalFlip(),
                tv.transforms.RandomHorizontalFlip(),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(mean=train_mean, std=train_std)
            ])
        else:
            self._transform = tv.transforms.Compose([
                tv.transforms.ToPILImage(),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(mean=train_mean, std=train_std)
            ])

    def __len__(self):
        return len(self.data)

    # returns the sample as a tuple: the image and the corresponding label.
    # This is memory efficient because all the images are not stored in the memory at once but read as required.
    def __getitem__(self, index):
        image = gray2rgb(imread(self.image_names[index]))
        image = self._transform(image)
        label = self.labels[index]

        sample = (image, label)
        return sample
