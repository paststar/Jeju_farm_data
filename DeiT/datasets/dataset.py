import torch
from torchvision import datasets
import torchvision.transforms as transforms
import torch.utils.data as torchdata
import numpy as np

def get_dataset(dataset_name, path='/database'):
    if dataset_name in ['스냅샷', '다중분광드론', '항공영상', 'RGB드론']:
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize([256, 256]),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize([224, 224]),
                transforms.CenterCrop((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
        if dataset_name == "RGB드론":
            tr = []
            tr.append(datasets.ImageFolder(path + '/' + dataset_name+'/1cm', data_transforms['train']))
            tr.append(datasets.ImageFolder(path + '/' + dataset_name+'/3cm', data_transforms['train']))
            tr.append(datasets.ImageFolder(path + '/' + dataset_name+'/5cm', data_transforms['train']))
            tr_dataset = torch.utils.data.ConcatDataset(tr)

            te = []
            te.append(datasets.ImageFolder(path + '/' + dataset_name + '/1cm', data_transforms['test']))
            te.append(datasets.ImageFolder(path + '/' + dataset_name + '/3cm', data_transforms['test']))
            te.append(datasets.ImageFolder(path + '/' + dataset_name + '/5cm', data_transforms['test']))
            print()
            te_dataset = torch.utils.data.ConcatDataset(te)
        else:
            tr_dataset = datasets.ImageFolder(path + '/' + dataset_name, data_transforms['train'])
            te_dataset = datasets.ImageFolder(path + '/' + dataset_name, data_transforms['test'])

    else:
        raise ValueError('Dataset %s not found!' % dataset_name)
    return tr_dataset, te_dataset

if __name__ == "__main__":
    batch_size = 16
    workers = 4

    path = '/home/bang/Desktop/jeju/dataset/제주주요작물_데이터'
    #trainset, testset = get_dataset("스냅샷", path = path)
    trainset, testset = get_dataset("RGB드론", path = path)

    indices = np.arange(0, len(trainset))
    np.random.seed(9696161)
    np.random.shuffle(indices)

    train_loader = torchdata.DataLoader(trainset, batch_size=batch_size, shuffle=False,
                                        num_workers=workers, pin_memory=True, drop_last=True,
                                        sampler=torch.utils.data.SubsetRandomSampler(indices[:int(len(trainset)*0.8)]))
    test_loader = torchdata.DataLoader(testset, batch_size=batch_size, shuffle=False,
                                           num_workers=workers, pin_memory=True, drop_last=False,
                                       sampler=torch.utils.data.SubsetRandomSampler(indices[int(len(trainset)*0.8)+1:] ))

    L = torch.tensor([0])
    for step, src_data in enumerate(train_loader):
        tgt_imgs, tgt_labels = src_data
        L=torch.cat((L, tgt_labels))
    print(torch.unique(L))


