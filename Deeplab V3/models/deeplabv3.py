import torch.nn
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models
from segdataset import *

def createDeepLabv3(outputchannels=1):
    """DeepLabv3 class with custom head
    Args:
        outputchannels (int, optional): The number of output channels
        in your dataset masks. Defaults to 1.
    Returns:
        model: Returns the DeepLabv3 model with the ResNet101 backbone.
    """
    model = models.segmentation.deeplabv3_resnet50(pretrained=True,
                                                    progress=True)
    model.classifier = DeepLabHead(2048, outputchannels)
    # Set the model in training mode
    model.train()
    return model

if __name__ == "__main__":
    batch_size = 8
    workers = 4

    train_transform = A.Compose(
        [
            A.PadIfNeeded(min_height=256, min_width=256),
            A.RandomCrop(256, 256),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    val_transform = A.Compose(
        [
            A.PadIfNeeded(min_height=256, min_width=256),
            A.CenterCrop(256, 256),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    dataset = SegmentationDataset(root='/home/bang/Desktop/jeju/dataset/제주 월동작물 자동탐지 드론 영상/Training',
                                  image_folder='Image1',mask_folder='Mask', transforms=train_transform)
    dataloder = torchdata.DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                     num_workers=workers, pin_memory=True)
    model = createDeepLabv3(outputchannels=3).cuda()
    #criterion = torch.nn.MSELoss(reduction='mean')
    criterion = torch.nn.CrossEntropyLoss()
    #criterion = torch.nn.BCELoss()

    for step, data in enumerate(dataloder):
        img, mask = data['image'].cuda(), data['mask'].cuda().long()
        outputs = model(img)
        print(img.type(), mask.type(),outputs['out'].type())
        loss = criterion(outputs['out'], mask)
        print(loss)
