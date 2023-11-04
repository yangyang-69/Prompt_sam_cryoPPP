import torch
import torch.nn as nn


class UNet_model(nn.Module):

    def __init__(self, in_channels=3, out_channels=3):
        '''
        initialize the unet
        '''
        super(UNet_model, self).__init__()
        ## con0_conv1_pool1
        self.encode1 = nn.Sequential(
            nn.Conv2d(in_channels, 48, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(2))
        ##conv2_pool2
        self.encode2 = nn.Sequential(
            nn.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(2))

        ##conv6_upsample5
        self.encode3 = nn.Sequential(
            nn.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.ConvTranspose2d(48, 48, 3, stride=2, padding=1, output_padding=1))

        #  deconv2a_2b_upsample1
        self.decode2 = nn.Sequential(
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1))
        ## deconv1a_1b
        self.decode1 = nn.Sequential(
            nn.Conv2d(96 + in_channels, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1))
        ## output layer
        self.output_layer = nn.Conv2d(32, out_channels, 3, stride=1, padding=1)

        # self.final = nn.LeakyReLU(negative_slope=0.1)

        ## initialize weight
        self._init_weights()

    def forward(self, x):
        '''
        forward function
        '''
        pool1 = self.encode1(x)
        pool2 = self.encode2(pool1)

        upsample2 = self.encode3(pool2)

        concat2 = torch.cat((upsample2, pool1), dim=1)
        upsample1 = self.decode2(concat2)
        concat1 = torch.cat((upsample1, x), dim=1)
        umsample0 = self.decode1(concat1)
        output = self.output_layer(umsample0)

        # output = self.final(output)

        return output

    def _init_weights(self):
        """Initializes weights using He et al. (2015)."""

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                nn.init.constant_(m.bias.data, 0)
