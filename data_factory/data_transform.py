from torchvision import transforms
from pl_bolts.transforms.self_supervised import Patchify
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

try:
    from torchvision import transforms
except ImportError:
    warn('You want to use `torchvision` which is not installed yet,'  # pragma: no-cover
         ' install it with `pip install torchvision`.')
    _TORCHVISION_AVAILABLE = False
else:
    _TORCHVISION_AVAILABLE = True


def data_transform(name, size):
    name = name.strip().split('+')
    name = [n.strip() for n in name]
    transform = []
    
    # Loading Method:
    if "resize_random_crop" in name:
        transform.extend([
            transforms.Resize(int(size * 8. / 7.)), # 224 -> 256
            transforms.RandomCrop(size),
            transforms.RandomHorizontalFlip(0.5)
            ])
    elif "resize" in name:
        transform.extend([
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(0.5)
            ])
    else:
        # "auto"
        transform.extend([
            transforms.Resize(size),
            transforms.CenterCrop(size)
            ])
    
    if "colorjitter" in name:
        transform.append(
            transforms.ColorJitter(brightness=0.4, saturation=0.4, hue=0.2))

    transform.extend([transforms.ToTensor(), normalize])
    transform = transforms.Compose(transform)
    return transform
    

class TrainTransforms():
    def __init__(self, patch_size=32, overlap=16):
        """
        Transforms used for CPC:

        Args:

            patch_size: size of patches when cutting up the image into overlapping patches
            overlap: how much to overlap patches

        Transforms::

            random_flip
            transforms.ToTensor()
            normalize
            Patchify(patch_size=patch_size, overlap_size=patch_size // 2)

        Example::

            # in a regular dataset
            Imagenet(..., transforms=CPCTrainTransformsImageNet128())

            # in a DataModule
            module = ImagenetDataModule(PATH)
            train_loader = module.train_dataloader(batch_size=32, transforms=CPCTrainTransformsImageNet128())
        """
        if not _TORCHVISION_AVAILABLE:
            raise ImportError('You want to use `transforms` from `torchvision` which is not installed yet.')

        # image augmentation functions
        self.patch_size = patch_size
        self.overlap = overlap
        self.flip_lr = transforms.RandomHorizontalFlip(p=0.5)
        rand_crop = transforms.RandomResizedCrop(224, scale=(0.3, 1.0), ratio=(0.7, 1.4), interpolation=3)
        col_jitter = transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.25)

        post_transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            # Patchify(patch_size=patch_size, overlap_size=overlap),
        ])

        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            rand_crop,
            col_jitter,
            rnd_gray,
            post_transform
        ])

    def __call__(self, inp):
        inp = self.flip_lr(inp)
        out1 = self.transforms(inp)
        return out1

class TestTransforms:
    def __init__(self, patch_size=32, overlap=16):
        """
        Transforms used for CPC:

        Args:

            patch_size: size of patches when cutting up the image into overlapping patches
            overlap: how much to overlap patches

        Transforms::

            random_flip
            transforms.ToTensor()
            normalize
            Patchify(patch_size=patch_size, overlap_size=patch_size // 2)

        Example::

            # in a regular dataset
            Imagenet(..., transforms=CPCEvalTransformsImageNet128())

            # in a DataModule
            module = ImagenetDataModule(PATH)
            train_loader = module.train_dataloader(batch_size=32, transforms=CPCEvalTransformsImageNet128())
        """
        if not _TORCHVISION_AVAILABLE:
            raise ImportError('You want to use `transforms` from `torchvision` which is not installed yet.')

        # image augmentation functions
        self.patch_size = patch_size
        self.overlap = overlap
        self.flip_lr = transforms.RandomHorizontalFlip(p=0.5)

        post_transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            Patchify(patch_size=patch_size, overlap_size=overlap),
        ])
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(224, interpolation=3),
            transforms.CenterCrop(128),
            post_transform
        ])

    def __call__(self, inp):
        inp = self.flip_lr(inp)
        out1 = self.transforms(inp)
        return out1
