from torch.utils.data import DataLoader
# from torch.utils.data.sampler import SubsetRandomSampler 
import torchvision.transforms as transforms
from ..torch_model.utils import ReScale, ToYUVTensor, ToRGBTensor, RandomRotate90, img2patch, patch2img
from ..torch_model.jpegmodules.quantize import Quantizer
from .base_dataset import BaseDataset

#//TODO complete this one.
class DeblockDataset(BaseDataset):
    def __init__(self, root, cropsize, Q, valid=True):
        super().__init__(root=root, task='deblocking', Q=Q)

        augment = [
        transforms.RandomCrop(size=cropsize),
        RandomRotate90(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        ]

        quantize = Quantizer(target_Q=Q)

        quantizing = [
        ToYUVTensor(),
        quantize,
        ToRGBTensor(),
        ]

        if not valid:
            self.target_transform = transforms.Compose(
                augment + [transforms.ToTensor(), ReScale(maxi=1)]
            )
            self.transform = transforms.Compose(
                augment + quantizing + [ReScale(maxi=255)]
            )
        else:
            # for eval
            self.target_transform = transforms.Compose(
                    [transforms.ToTensor(), ReScale(maxi=1),]
            )
            self.transform = transforms.Compose(
                quantizing + [ReScale(maxi=255)]
            )

def deblock_dataloader( dataroot,
                        cropsize,
                        batch_size, 
                        num_workers,
                        Q=None,
                        valid=False
                        ):

    dataset = DeblockDataset(dataroot, cropsize, Q, valid=valid)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            drop_last=(not valid),
                            shuffle=(not valid),
                            num_workers=num_workers
                            )

    data = {'loader':dataloader,
            'num':dataset.__len__()}

    return data
