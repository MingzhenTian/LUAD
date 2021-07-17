# 数据载入部分
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms as tf
from PIL import Image
import os


class MyDataset(Dataset):

    def __init__(self, root):
        self.data_info = self.get_img_info(root)

        self.transforms = tf.Compose([
            tf.RandomHorizontalFlip(),
            tf.RandomCrop(224, pad_if_needed = False,fill = 0, padding_mode='constant'),
            tf.ToTensor()
        ])

    def __getitem__(self, index):
        path_img, label = self.data_info[index]
        x = Image.open(path_img).convert('RGB')
        # x=np.array(x)
        x = self.transforms(x)
        return x, label

    def __len__(self):
        return len(self.data_info)

    @staticmethod
    def get_img_info(root):
        all = open('luad.csv').readlines()[1:]
        dic = {}
        for i in all:
            id = i.split(",")[2]
            state = i.split(',')[26]
            id = eval(id)
            state = eval(state)
            dic[id] = state
        data_info = list()
        label=0
        for root, dirs, files in os.walk(root):
            for name in files:
                path_img = os.path.join(root, name)
                id2 = name[0:12]
                if dic[id2] == 'Alive':
                    label = 1
                if dic[id2] == 'Dead':
                    label = 0
                data_info.append((path_img, int(label)))
        return data_info



def obtain_loader(train_root,test_root):
    train_set = MyDataset(root=train_root)
    test_set = MyDataset(root=test_root)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=0)
    return train_loader, test_loader
