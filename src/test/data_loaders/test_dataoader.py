from argparse import ArgumentParser
from time import perf_counter
from typing import List

import torch
from torch.utils.data import DataLoader
from torch.utils.data import SequentialSampler,RandomSampler,BatchSampler
try:
    import intel_extension_for_pytorch
except ModuleNotFoundError as err:
    print(err)

import datasets


def main():
    """Test performance of various DataSets and DataLoaders
    """

    parser = ArgumentParser(description='SmartRedis Data Producer')
    parser.add_argument('--batch_size', default=8192, type=int, help='Batch size')
    parser.add_argument('--precision', default='fp32', type=str, help='Precision of data (fp32,fp64,bf16)')
    parser.add_argument('--device', default='xpu', type=str, help='Device to run dataloader on (xpu,cpu)')
    parser.add_argument('--datasets', default=['MiniBatchDataset','MiniBatchDataset_BatchSampler'], type=List[str], help='List of datasets to test')
    args = parser.parse_args()

    # Generate random data
    if (args.precision == "fp32"):
        dtype = torch.float32
    elif (args.precision == "fp64"):
        dtype = torch.float64
    elif (args.precision == "bf16"):
        dtype = torch.bfloat16
    data = torch.rand((args.batch_size*200,12)).type(dtype)

    # Set device
    device = torch.device(args.device)
    #data = data.to(device)
    
    # Loop over input datasets and measure DataLoader time for 1 epoch
    print('Testing with DataSets: ',*args.datasets,'\n', flush=True)
    # Try:
        # - pin_memory=True - should be faster for GPU training
        # - num_workers > 1 - enables multi-process data loading
        # - prefetch_factor >1 - enables pre-fetching of data
    for dataset in args.datasets:
        dataset_name = dataset.split('_')[0]
        sampler_name = dataset.split('_')[1] if len(dataset.split('_'))>1 else None
        
        data_set = getattr(datasets, dataset_name)(data)
        #print(f'Length of dataset: {data_set.__len__()}')
        #print(f'Item of dataset: {data_set.__getitem__(0)}')
        
        sampler = None; shuffle = True
        if sampler_name=='BatchSampler':
            #sampler = BatchSampler(SequentialSampler(data_set), batch_size=args.batch_size, drop_last=False)
            sampler = BatchSampler(RandomSampler(data_set), batch_size=args.batch_size, drop_last=False)
            shuffle = False

        data_loader = DataLoader(data_set,shuffle=shuffle, batch_size=args.batch_size,
                                 sampler=sampler,
                                 #pin_memory=True,
                                 #num_workers=2,
                                 #prefetch_factor=1,
                                )
        tic = perf_counter()
        for batch_id, batch in enumerate(data_loader):
            if args.device=='xpu': batch = batch.to(device)
        toc = perf_counter()
        print(f'DataLoader time for {dataset}: {toc-tic:>.4f} sec', flush=True)


if __name__ == "__main__":
    main()
