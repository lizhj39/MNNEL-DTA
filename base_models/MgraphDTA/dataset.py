import torch
from torch_geometric.data import InMemoryDataset

class GNNDataset(InMemoryDataset):

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['data.csv']

    @property
    def processed_file_names(self):
        return ['processed_data.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def process(self):
        pass

# if __name__ == "__main__":
#     dataset1 = GNNDataset('data/davis')
#     dataset2 = GNNDataset('data/kiba')


