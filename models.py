import torch
from torch.nn import Module


class AVModel(Module):
    def __init__(self, vision_encoder, audio_encoder, projection_head):
        """Audio-visual model"""

        super(AVModel, self).__init__()
        self.vision_encoder = vision_encoder
        self.audio_encoder = audio_encoder
        self.projection_head = projection_head

    def forward(self, v, a):
        """Inputs are video and audio frames respectively"""

        # pass inputs through the encoders
        v = self.vision_encoder(v)
        a = self.audio_encoder(a)

        # concatenate the embeddings and pass through projection head
        av = torch.cat((v, a), dim=1)
        av = self.projection_head(av)

        return av