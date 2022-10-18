import torch
from torch import nn

class GENIUS(nn.Module):
    def __init__(self, backbone, input):
        super().__init__()
        self.input_layer = input
        self.backbone = backbone

    def store_backbone(self, ep, optimizer, loss,image_type, folder):
        path_input = "/home/mateo/pytorch_docker/TCGA_GenomeImage/src/modeling/checkpoints/input__{}_{}.pb".format(
                                                                                                                  image_type,
                                                                                                                  folder.replace(
                                                                                                                      "/",
                                                                                                                      "_"))
        path_backbone = "/home/mateo/pytorch_docker/TCGA_GenomeImage/src/modeling/checkpoints/backbone__{}_{}.pb".format(image_type,
                                                                                                     folder.replace("/",
                                                                                                                    "_"))
        torch.save({
            'epoch': ep,
            'model_state_dict': self.input_layer.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }, path_input)
        torch.save({
            'epoch': ep,
            'model_state_dict': self.backbone.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }, path_backbone)
        return path_backbone, self.backbone.state_dict(),  path_input, self.input_layer.state_dict(),optimizer.state_dict()

    def forward(self, x):
        x = self.input_layer(x)
        embedding = self.backbone(x)
        return embedding