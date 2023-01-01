import torch
import torch.nn as nn
import torch.nn.functional as F

AVAI_HEADS = ['linear', 'adapter']


class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4, residual_ratio=0.2):
        super(Adapter, self).__init__()
        self.residual_ratio = residual_ratio
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        a = self.fc(x)
        x = self.residual_ratio * a + (1 - self.residual_ratio) * x
        return x


def get_zero_shot_weights(text_dataset, num_classes, in_features):
    # Caveat: Only support text_dataset with 1-D text features. 
    # Need to modify if you want to partial finetuning the text encoder
    weights = torch.zeros(num_classes, in_features)
    count = torch.zeros(num_classes)
    for i in range(len(text_dataset)):
        label = text_dataset.label_tensor[i]
        weights[label] += F.normalize(text_dataset.input_tensor[i], dim=0)
        count[label] += 1
    weights /= count.unsqueeze(1)
    # normalize the weights
    weights.data = F.normalize(weights, dim=1)
    return weights


def make_classifier_head(classifier_head,
                         clip_encoder,
                         classifier_init,
                         zeroshot_dataset,
                         bias=False):
    assert classifier_head in AVAI_HEADS
    if clip_encoder == 'ViT-B/16':
        in_features = 512
    elif clip_encoder == 'RN50':
        in_features = 1024

    num_classes = int(zeroshot_dataset.label_tensor.max()) + 1

    linear_head = nn.Linear(in_features, num_classes, bias=bias)
    if classifier_init == 'zeroshot':
        assert zeroshot_dataset.input_tensor.shape[1] == in_features
        linear_head.weight.data = get_zero_shot_weights(
            zeroshot_dataset, num_classes, in_features)
    
    if classifier_head == 'linear':
        head = linear_head
    elif classifier_head == 'adapter':
        adapter = Adapter(in_features, residual_ratio=0.2)
        head = nn.Sequential(
            adapter,
            linear_head
        )
    else:
        raise ValueError(f"Invalid head: {classifier_head}")
    return head, num_classes, in_features