import os
import random
import numpy as np
import torch
from PIL import Image
from PIL import ImageFilter
import torch.utils.data as data
import torchvision.transforms as transforms
from transforms_bag.autoaugment import ImageNetPolicy, CIFAR10Policy, SVHNPolicy
from transforms_bag.randaugment import RandAugmentMC


def _tolist(x):
    if isinstance(x, list):
        return x
    elif isinstance(x, (np.ndarray, torch.Tensor)):
        return x.tolist()
    else:
        raise TypeError

class TwoCropsTransform:

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class GaussianBlur(object):

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def folder_content_getter(folder_path):
    cate_names = list(np.sort(os.listdir(folder_path)))

    image_path_list = []
    image_cate_list = []

    for cate_name in cate_names:
        sub_folder_path = os.path.join(folder_path, cate_name)
        if os.path.isdir(sub_folder_path):
            image_names = list(np.sort(os.listdir(sub_folder_path)))
            for image_name in image_names:
                image_path = os.path.join(sub_folder_path, image_name)
                image_path_list.append(image_path)
                image_cate_list.append(cate_names.index(cate_name))

    return image_path_list, image_cate_list

class EvalDataset(data.Dataset):
    def __init__(self,
                 datasetA_dir,
                 datasetB_dir):

        self.datasetA_dir = datasetA_dir
        self.datasetB_dir = datasetB_dir

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        self.transform = transforms.Compose([
                         transforms.Resize(256),
                         transforms.CenterCrop(224),
                         transforms.ToTensor(),
                         normalize,
                     ])

        self.image_paths_A, self.image_cates_A = folder_content_getter(datasetA_dir)
        self.image_paths_B, self.image_cates_B = folder_content_getter(datasetB_dir)

        self.domainA_size = len(self.image_paths_A)
        self.domainB_size = len(self.image_paths_B)

    def __getitem__(self, index):
        index_A = np.mod(index, self.domainA_size)
        index_B = np.mod(index, self.domainB_size)

        image_path_A = self.image_paths_A[index_A]
        image_path_B = self.image_paths_B[index_B]

        image_A = self.transform(Image.open(image_path_A).convert('RGB'))
        image_B = self.transform(Image.open(image_path_B).convert('RGB'))

        target_A = self.image_cates_A[index_A]
        target_B = self.image_cates_B[index_B]
        return image_A, index_A, target_A, image_B, index_B, target_B

    def __len__(self):

        return max(self.domainA_size, self.domainB_size)


class TrainDataset(data.Dataset):
    def __init__(self,
                 datasetA_dir,
                 datasetB_dir,
                 aug_plus,
                 indices= None,
                 labels = None,
                 return_label =False
                 ):

        self.datasetA_dir = datasetA_dir
        self.datasetB_dir = datasetB_dir

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        if aug_plus:
            self.transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
                    transforms.RandomApply([
                        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  
                    ], p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize
                ]
            )
        self.image_paths_A, self.image_cates_A = folder_content_getter(datasetA_dir)
        self.image_paths_B, self.image_cates_B = folder_content_getter(datasetB_dir)

        if indices is not None:
            self.image_paths_A = [self.image_paths_A[i] for i in indices['A']]
            self.image_paths_B = [self.image_paths_B[i] for i in indices['B']]
            self.image_cates_A = [self.image_cates_A[i] for i in indices['A']]
            self.image_cates_B = [self.image_cates_B[i] for i in indices['B']]
        self.label_A = None
        self.label_B = None
        if labels is not None:
            self.label_A = _tolist(labels['A']) 
            self.label_B = _tolist(labels['B'])
        self.domainB_size = len(self.image_paths_B)
        self.domainA_size = len(self.image_paths_A)
    def __getitem__(self, index):

        if index >= self.domainA_size:
            index_A = random.randint(0, self.domainA_size - 1)
        else:
            index_A = index

        if index >= self.domainB_size:
            index_B = random.randint(0, self.domainB_size - 1)
        else:
            index_B = index

        image_path_A = self.image_paths_A[index_A]
        image_path_B = self.image_paths_B[index_B]

        x_A = Image.open(image_path_A).convert('RGB')
   
        q_A = self.transform(x_A)
        k_A = self.transform(x_A)

        x_B = Image.open(image_path_B).convert('RGB')
        q_B = self.transform(x_B)
        k_B = self.transform(x_B)


        target_A = self.image_cates_A[index_A]
        target_B = self.image_cates_B[index_B]        
        if self.label_A is not None:
            return [q_A, k_A], self.label_A[index_A], index_A, [q_B, k_B], self.label_B[index_B], index_B
        return [q_A, k_A], index_A, [q_B, k_B], index_B, target_A, target_B

    def __len__(self):

        return max(self.domainA_size, self.domainB_size)
    
class aug_dataset(data.Dataset):
    def __init__(self,dataset_dir,views='ws'):
        self.dataset_dir = dataset_dir
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        self.views = views
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
                    transforms.RandomApply([
                        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                    ], p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize
                ]
            )
        self.strong1 = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            RandAugmentMC(n=2, m=10),    # strong aug: fixmatch
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        self.strong2 = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            ImageNetPolicy(),    
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        self.weak = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),      
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        self.image_paths, self.image_cates = folder_content_getter(dataset_dir)
        self.domain_size = len(self.image_paths)
    
    
    def __getitem__(self, index):
    
        if index >= self.domain_size:
            index = random.randint(0, self.domain_size - 1)
        else:
            index = index
            
        image_path = self.image_paths[index]
        img = Image.open(image_path).convert('RGB')
        ret_views = []
        if self.views == 'ss':
            ret_views.append(self.strong1(img))
            ret_views.append(self.strong2(img))
        elif self.views == 'ws':
            ret_views.append(self.strong2(img))
            ret_views.append(self.weak(img))
        else:
            ret_views.append(self.transform(img))
        
        if len(ret_views) > 1:
            return ret_views,index
        else:
            return ret_views[0],index    
        
    def __len__(self):
        return self.domain_size
