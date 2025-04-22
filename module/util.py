import torch
import torch.nn as nn
import torchxrayvision as xrv
import torchvision.models as models

def get_model(model_tag, num_classes, pretrained_imagenet=False, pretrained_xrv=False, mlp_classifier=False, num_heads=None, DFA=False):
    if model_tag == "DenseNet121":
        if pretrained_xrv:
            model = xrv.models.DenseNet(weights="densenet121-res224-all")
        elif pretrained_imagenet:
            model = models.densenet121(pretrained=True)
        else:
            model = models.densenet121(pretrained=False)
        if mlp_classifier:
            if not DFA:
                if num_heads == None:
                    model.classifier = nn.Sequential(nn.Linear(1024, 512),
                                                    nn.ReLU(),
                                                    nn.Linear(512, 512),
                                                    nn.ReLU(),
                                                    nn.Linear(512, num_classes))
                elif num_heads >= 1:
                    model.classifiers = nn.ModuleList([nn.Sequential(nn.Linear(1024, 512),
                                                                nn.ReLU(),
                                                                nn.Linear(512, 512),
                                                                nn.ReLU(),
                                                                nn.Linear(512, num_classes)) for i in range(num_heads)])
            else:
                model.classifier = nn.Sequential(nn.Linear(2048, 512),
                                                nn.ReLU(),
                                                nn.Linear(512, 512),
                                                nn.ReLU(),
                                                nn.Linear(512, num_classes))
        else:
            if not DFA:
                model.classifier = nn.Linear(1024, num_classes)
            else:
                model.classifier = nn.Linear(2048, num_classes)
        model.upsample = None

        # for param in model.parameters():
        #     param.requires_grad = False
        # for param in model.classifier.parameters():
        #     param.requires_grad = True

        return model
    
    elif model_tag == 'ResNet18':
        if pretrained_imagenet:
            model = models.resnet18(pretrained=True)
        else:
            model = models.resnet18(pretrained=False)

        model.features = nn.Sequential(model.conv1, model.bn1, model.relu, 
                                model.maxpool, model.layer1, model.layer2, 
                                model.layer3, model.layer4, model.avgpool)
        
        if mlp_classifier:
            if not DFA:
                if num_heads == None:
                    model.fc = nn.Sequential(nn.Linear(512, 512),
                                            nn.ReLU(),
                                            nn.Linear(512, 512),
                                            nn.ReLU(),
                                            nn.Linear(512, num_classes))
                elif num_heads >= 1:
                    model.classifiers = nn.ModuleList([nn.Sequential(nn.Linear(512, 512),
                                                                    nn.ReLU(),
                                                                    nn.Linear(512, 512),
                                                                    nn.ReLU(),
                                                                    nn.Linear(512, num_classes)) for i in range(num_heads)])
            else:
                model.fc = nn.Sequential(nn.Linear(1024, 512),
                                            nn.ReLU(),
                                            nn.Linear(512, 512),
                                            nn.ReLU(),
                                            nn.Linear(512, num_classes))
            # else:
            #     model.fc = nn.Sequential(nn.Linear(512, 512),
            #                             nn.ReLU(),
            #                             nn.Linear(512, 512),
            #                             nn.ReLU(),
            #                             nn.Linear(512, num_classes))
        else:
            if not DFA:
                model.fc = nn.Linear(512, num_classes)
            else:
                model.fc = nn.Linear(1024, num_classes)
            
        return model
    
    else:
        raise NotImplementedError
    


class MultiDimAverageMeter(object):
    def __init__(self, dims):
        self.dims = dims
        self.cum = torch.zeros(np.prod(dims))
        self.cnt = torch.zeros(np.prod(dims))
        self.idx_helper = torch.arange(np.prod(dims), dtype=torch.long).reshape(
            *dims
        )

    def add(self, vals, idxs):
        flattened_idx = torch.stack(
            [self.idx_helper[tuple(idxs[i])] for i in range(idxs.size(0))],
            dim=0,
        )
        self.cum.index_add_(0, flattened_idx, vals.view(-1).float())
        self.cnt.index_add_(
            0, flattened_idx, torch.ones_like(vals.view(-1), dtype=torch.float)
        )
        
    def get_mean(self):
        return (self.cum / self.cnt).reshape(*self.dims)

    def reset(self):
        self.cum.zero_()
        self.cnt.zero_()