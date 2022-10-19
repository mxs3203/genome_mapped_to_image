import torch
from torch import nn

class GENIUS(nn.Module):
    def __init__(self, backbone, input):
        super().__init__()
        self.input_layer = input
        self.backbone = backbone
        self.predictor = nn.Sequential(
            nn.Linear(5000, 1024), nn.ReLU(),
            nn.Linear(1024, 1024), nn.ReLU(),
            nn.Linear(1024, 1024), nn.ReLU(),
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, 2)
        )
        self.softmax = nn.Softmax(dim=1)

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

    def store_predictor(self, ep, optimizer,loss, image_type,folder):
        path_predictor = "/home/mateo/pytorch_docker/TCGA_GenomeImage/src/modeling/checkpoints/predictor_{}_{}.pb".format(
            image_type,
            folder.replace("/",
                           "_"))
        torch.save({
            'epoch': ep,
            'model_state_dict': self.backbone.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }, path_predictor)
        return path_predictor
    def forward(self, x):
        x = self.input_layer(x)
        embedding = self.backbone(x)
        y = self.softmax(self.predictor(embedding))
        return y