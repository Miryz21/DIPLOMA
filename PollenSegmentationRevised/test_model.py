import torch

import utils
from engine import evaluate
from train_model import get_model_instance_segmentation, SampleDataset


if __name__ == '__main__':
    test_root = r'Data/CVAT_dataset'
    model_root = r'Save/models/InSegModel_new_wtl'

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = get_model_instance_segmentation(num_classes=2, pre_trained=False)
    model.load_state_dict(torch.load(model_root, map_location=device))
    model.to(device)
    model.eval()
    

    real_dataset = SampleDataset(root=test_root)

    seed = 28
    generator = torch.Generator().manual_seed(seed)

    val_dataset, test_dataset = torch.utils.data.random_split(
        real_dataset,
        [124, len(real_dataset)-124],
        generator=generator 
    )

    data_loader_val = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=utils.collate_fn)
    test_loader =  torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=utils.collate_fn)

    evaluate(model, test_loader, device)
