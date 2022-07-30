import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
import yaml
from torch.utils.data import Dataset


class DataSetManual(Dataset):
    def __init__(self,
                 data_csv_income,
                 images_path: str = './',
                 transform=None):
        super().__init__()
        self.transform = transform
        self.images_path = images_path
        self.data_csv_income = data_csv_income
        self.name = data_csv_income['image']
        self.label = data_csv_income['label']
        vpa = data_csv_income['label']
        self.materials_name_func, self.materials_list_func = self.material_founder()
        self.materials = {}
        self.dict_labels = {}
        self.dict_materials = {}
        for i in range(len(vpa)):
            if vpa[i] not in self.dict_labels:
                self.dict_labels[vpa[i]] = len(self.dict_labels)
        self.list_labels = list(self.dict_labels)
        for i in range(len(self.materials_name_func)):
            if self.materials_name_func[i] not in self.dict_materials:
                self.dict_materials[self.materials_name_func[i]] = len(self.dict_materials)
        self.list_materials = list(self.dict_materials)

    def load_classes(self, src: str = None):
        if src is not None:
            with open(src, 'r') as read:
                classes = yaml.full_load(read)
                self.dict_labels = classes[0]
                self.dict_materials = classes[1]

    def material_founder(self):

        type_cl = []
        type_nl = []
        for i in range(self.__len__()):
            if self.label[i] == 'Not sure':
                type_cl.append('Not sure')
                type_nl.append(0)

            if self.label[i] == 'T-Shirt':
                type_cl.append('thread')
                type_nl.append(1)

            if self.label[i] == 'Shoes':
                type_cl.append('Canvas / rubber / plastics')
                type_nl.append(2)

            if self.label[i] == 'Shorts':
                type_cl.append('Denim')
                type_nl.append(3)

            if self.label[i] == 'Shirt':
                type_cl.append('Poplin')
                type_nl.append(4)

            if self.label[i] == 'Pants':
                type_cl.append('cotton / wool')
                type_nl.append(5)

            if self.label[i] == 'Skirt':
                type_cl.append('Cotton / Linen')
                type_nl.append(6)

            if self.label[i] == 'Other':
                type_cl.append('Other')
                type_nl.append(7)

            if self.label[i] == 'Top':
                type_cl.append('UnKnown-Top')
                type_nl.append(8)

            if self.label[i] == 'Outwear':
                type_cl.append('Polyester ')
                type_nl.append(9)

            if self.label[i] == 'Dress':
                type_cl.append('fabrics')
                type_nl.append(10)

            if self.label[i] == 'Body':
                type_cl.append('UnKnown-Body')
                type_nl.append(11)

            if self.label[i] == 'Long-sleeve':
                type_cl.append(' polyester / cotton')
                type_nl.append(12)

            if self.label[i] == 'Undershirt':
                type_cl.append('cotton / linen')
                type_nl.append(13)

            if self.label[i] == 'Hat':
                type_cl.append('Polyester / straw')
                type_nl.append(14)

            if self.label[i] == 'Polo':
                type_cl.append('polyester / rayon')
                type_nl.append(15)

            if self.label[i] == 'Blouse':
                type_cl.append('Georgette / Crepe')
                type_nl.append(16)

            if self.label[i] == 'Hoodie':
                type_cl.append('jeans / tartan')
                type_nl.append(17)

            if self.label[i] == 'Skip':
                type_cl.append('Skip')
                type_nl.append(18)

            if self.label[i] == 'Blazer':
                type_cl.append('Worsted Wool / Flannel / Fresco')
                type_nl.append(19)
        return type_cl, type_nl

    def __len__(self):
        return len(self.name)

    def __getitem__(self, item):
        image = plt.imread(f'{self.images_path}/{self.name[item]}.jpg')
        image = torch.FloatTensor(image)
        image = image.permute(2, 0, 1)
        if image.shape[0] == 4:
            image = image[:3, :, :]

        image = T.Normalize((0, 0, 0), (1, 1, 1))(image)
        image = T.ToPILImage()(image)

        image = T.Resize((224, 224))(image)
        image = T.ToTensor()(image)
        label = self.label[item]
        target = torch.zeros(40)
        target[self.dict_labels[label]] = 1
        target[self.materials_list_func[item] + 20] = 1
        ignorance = {
            'Not sure',
            'Skip',
            'Top',
            'Other'
        }
        return image, target
