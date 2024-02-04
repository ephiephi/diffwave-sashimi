import torch
from torch.utils.data.distributed import DistributedSampler
from .sc import SpeechCommands
from .mel2samp import Mel2Samp
from .uni import Unified


def dataloader(dataset_cfg, batch_size, num_gpus, unconditional=True):
    # TODO would be nice if unconditional was decoupled from dataset
    print("##########")
    dataset_name = dataset_cfg.pop("_name_")
    print(dataset_name)
    if dataset_name == "sc09":
        assert unconditional
        print("dataset_cfg.data_path: ", dataset_cfg.data_path)
        dataset = SpeechCommands(dataset_cfg.data_path)
    elif dataset_name == "unified":
        assert unconditional
        dataset = Unified(dataset_cfg.data_path)
    elif dataset_name == "ljspeech":
        assert not unconditional
        dataset = Mel2Samp(**dataset_cfg)
    dataset_cfg["_name_"] = dataset_name  # Restore

    # distributed sampler
    train_sampler = DistributedSampler(dataset) if num_gpus > 1 else None

    trainloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=2,
        pin_memory=False,
        drop_last=True,
    )
    return trainloader
