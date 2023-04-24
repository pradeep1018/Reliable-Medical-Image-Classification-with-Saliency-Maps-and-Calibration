import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from skimage import io
import medmnist
from medmnist import INFO

class CheXpertData(Dataset):
    def __init__(self, label_path, mode='train'):
        self._label_header = None
        self._image_paths = []
        self._labels = []
        self._mode = mode
        self.transform = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST),
             transforms.Grayscale(num_output_channels=3),
             transforms.ToTensor(),
             transforms.Normalize(mean=[.5], std=[.5])])
        self.dict = [{'1.0': '1', '': '0', '0.0': '0', '-1.0': '0'},
                     {'1.0': '1', '': '0', '0.0': '0', '-1.0': '1'}, ]
        with open(label_path) as f:
            header = f.readline().strip('\n').split(',')
            self._label_header = [
                header[7],
                header[10],
                header[11],
                header[13],
                header[15]]
            c1 = 0 
            c2 = 0
            c3 = 0
            c4 = 0
            c5 = 0
            for line in f:
                labels = []
                fields = line.strip('\n').split(',')
                image_path = fields[0]
                for index, value in enumerate(fields[5:]):
                    if index == 5 or index == 8:
                        labels.append(self.dict[1].get(value))
                    elif index == 2 or index == 6 or index == 10:
                        labels.append(self.dict[0].get(value))
                    """
                    if index == 2 and self.dict[0].get(value) == '1':
                        c1+=1
                    elif index == 5 and self.dict[1].get(value) == '1':
                        c2+=1
                    elif index == 6 and self.dict[0].get(value) == '1':
                        c3+=1
                    elif index == 8 and self.dict[1].get(value) == '1':
                        c4+=1
                    elif index == 10 and self.dict[0].get(value) == '1':
                        c5+=1
                    """
                labels = list(map(int, labels))
                self._image_paths.append(image_path)
                self._labels.append(labels)
                """
                if c1<=9000 and labels[0]==1:
                    self._image_paths.append(image_path)
                    self._labels.append(labels)
                elif c2<=9000 and labels[1]==1:
                    self._image_paths.append(image_path)
                    self._labels.append(labels)
                elif c3<=9000 and labels[2]==1:
                    self._image_paths.append(image_path)
                    self._labels.append(labels)
                elif c4<=9000 and labels[3]==1:
                    self._image_paths.append(image_path)
                    self._labels.append(labels)
                elif c5<=9000 and labels[4]==1:
                    self._image_paths.append(image_path)
                    self._labels.append(labels)
                """
        self._num_image = len(self._image_paths)

    def __len__(self):
        return self._num_image

    def __getitem__(self, idx):
        image = io.imread(self._image_paths[idx])
        image = self.transform(image)
        labels = np.array(self._labels[idx]).astype(np.float32)
        path = self._image_paths[idx]

        if self._mode == 'train' or self._mode == 'val':
            return (image, labels)
        elif self._mode == 'test':
            return (image, path)
        else:
            raise Exception('Unknown mode : {}'.format(self._mode))

def get_data_medmnist(data_flag, resize=True, download=True, as_rgb=True):
    info = INFO[data_flag]
    task = info['task']
    n_classes = len(info['label'])
    DataClass = getattr(medmnist, info['python_class'])

    if resize:
        data_transform = transforms.Compose(
            [transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST),
             #transforms.Grayscale(num_output_channels=3),
             transforms.ToTensor(),
             transforms.Normalize(mean=[.5], std=[.5])])
    else:
        data_transform = transforms.Compose(
            [#transforms.Grayscale(num_output_channels=3),
             transforms.ToTensor(),
             transforms.Normalize(mean=[.5], std=[.5])])

    train_dataset = DataClass(split='train', transform=data_transform, download=download, as_rgb=as_rgb)
    val_dataset = DataClass(split='val', transform=data_transform, download=download, as_rgb=as_rgb)

    return train_dataset, val_dataset, n_classes