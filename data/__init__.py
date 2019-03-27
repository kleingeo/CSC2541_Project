import torch.utils.data
from data.base_data_loader import BaseDataLoader


def CreateDataLoader(opt):
    data_loader = CustomDatasetDataLoader()
    print(data_loader.name())
    data_loader.initialize(opt)
    return data_loader


def CreateDataset(opt):
    dataset = None
    if opt.dataset_mode == 'aligned':
        from data.aligned_dataset import AlignedDataset
        dataset = AlignedDataset()
    elif opt.dataset_mode == 'unaligned':
        from data.unaligned_dataset import UnalignedDataset
        dataset = UnalignedDataset()
    elif opt.dataset_mode == 'single':
        from data.single_dataset import SingleDataset
        dataset = SingleDataset()
    elif opt.dataset_mode == 'aligned3d':
        from data.aligned3d_dataset import Aligned3dDataset
        dataset = Aligned3dDataset()
    elif opt.dataset_mode == 'aligned3dto2d':
        from data.aligned_3d_to_2d_dataset import Aligned3dTo2dDataset
        dataset = Aligned3dTo2dDataset()
    elif opt.dataset_mode == 'aligned3dtoimage':
        from data.aligned_3d_to_image_dataset import Aligned3dToImageDataset
        dataset = Aligned3dToImageDataset()
    elif opt.dataset_mode == 'unaligned3dtoimage':
        from data.unaligned_3d_to_image_dataset import Unaligned3dToImageDataset
        dataset = Unaligned3dToImageDataset()
    elif opt.dataset_mode == 'aligned2dtoimage':
        from data.aligned_2d_to_image_dataset import Aligned2dToImageDataset
        dataset = Aligned2dToImageDataset()
    elif opt.dataset_mode == 'unaligned2dtoimage':
        from data.unaligned_2d_to_image_dataset import Unaligned2dToImageDataset
        dataset = Unaligned2dToImageDataset()

    elif opt.dataset_mode == 'aligned2dnpy':
        from data.aligned_2d_npy_dataset import Aligned2dNpyDataset
        dataset = Aligned2dNpyDataset()

    elif opt.dataset_mode == 'aligned2dnpyfixed':
        from data.aligned_2d_npy_fixed_dataset import Aligned2dNpyFixedDataset
        dataset = Aligned2dNpyFixedDataset()

    else:
        raise ValueError("Dataset [%s] not recognized." % opt.dataset_mode)

    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset


class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads))

    def load_data(self):
        return self

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batchSize >= self.opt.max_dataset_size:
                break
            yield data
