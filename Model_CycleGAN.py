# Modified from junyanz/pytorch-CycleGAN-and-pix2pix for 3D volumetric data
# Original: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
import torch
import torch.nn as nn
from torch.nn import init
import functools


###############################################################################
# Helper Functions
###############################################################################

def get_norm_layer_3d(norm_type="instance"):
    """Return a 3D normalization layer"""
    if norm_type == "batch":
        norm_layer = functools.partial(nn.BatchNorm3d, affine=True, track_running_stats=True)
    elif norm_type == "instance":
        norm_layer = functools.partial(nn.InstanceNorm3d, affine=False, track_running_stats=False)
    elif norm_type == "none":
        def norm_layer(x):
            return nn.Identity()
    else:
        raise NotImplementedError("normalization layer [%s] is not found" % norm_type)
    return norm_layer


def init_weights(net, init_type="normal", init_gain=0.02):
    """Initialize network weights"""
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (classname.find("Conv") != -1 or classname.find("Linear") != -1):
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            else:
                raise NotImplementedError("initialization method [%s] is not implemented" % init_type)
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find("BatchNorm3d") != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print("initialize network with %s" % init_type)
    net.apply(init_func)


###############################################################################
# Generator Classes
###############################################################################

class ResnetGenerator3D(nn.Module):
    """3D Resnet-based generator for volumetric data"""

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm3d, use_dropout=False, n_blocks=6, padding_type="reflect"):
        """
        Construct a 3D Resnet-based generator
        
        Parameters:
            input_nc (int)      -- number of input channels
            output_nc (int)     -- number of output channels
            ngf (int)           -- number of filters in the last conv layer
            norm_layer          -- normalization layer (3D)
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- number of ResNet blocks
            padding_type (str)  -- padding layer type: reflect | replicate | zero
        """
        assert n_blocks >= 0
        super(ResnetGenerator3D, self).__init__()
        
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

        model = [nn.ReplicationPad3d(3), 
                 nn.Conv3d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias), 
                 norm_layer(ngf), 
                 nn.ReLU(True)]

        # Downsampling
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv3d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        # ResNet blocks
        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock3D(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, 
                                   use_dropout=use_dropout, use_bias=use_bias)]

        # Upsampling
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose3d(ngf * mult, int(ngf * mult / 2), 
                                        kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        
        model += [nn.ReplicationPad3d(3)]
        model += [nn.Conv3d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class ResnetBlock3D(nn.Module):
    """Define a 3D Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """
        Initialize the Resnet block
        
        A resnet block is a conv block with skip connections
        """
        super(ResnetBlock3D, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a 3D convolutional block"""
        conv_block = []
        p = 0
        
        if padding_type == "reflect":
            # 3D reflection padding not available, use replication
            conv_block += [nn.ReplicationPad3d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad3d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)

        conv_block += [nn.Conv3d(dim, dim, kernel_size=3, padding=p, bias=use_bias), 
                       norm_layer(dim), 
                       nn.ReLU(True)]
        
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReplicationPad3d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad3d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)
        
        conv_block += [nn.Conv3d(dim, dim, kernel_size=3, padding=p, bias=use_bias), 
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)
        return out


###############################################################################
# Discriminator Classes
###############################################################################

class NLayerDiscriminator3D(nn.Module):
    """Defines a 3D PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm3d):
        """
        Construct a 3D PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input volumes
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator3D, self).__init__()
        
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

        kw = 3  # Changed from 4 to 3 for smaller volumes
        padw = 1
        sequence = [nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), 
                    nn.LeakyReLU(0.2, True)]
        
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv3d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


###############################################################################
# Loss Classes
###############################################################################

class GANLoss(nn.Module):
    """Define different GAN objectives"""

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """
        Initialize the GANLoss class

        Parameters:
            gan_mode (str) - - type of GAN objective: vanilla | lsgan | wgangp
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label for a fake image
        """
        super(GANLoss, self).__init__()
        self.register_buffer("real_label", torch.tensor(target_real_label))
        self.register_buffer("fake_label", torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        
        if gan_mode == "lsgan":
            self.loss = nn.MSELoss()
        elif gan_mode == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ["wgangp"]:
            self.loss = None
        else:
            raise NotImplementedError("gan mode %s not implemented" % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input"""
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and ground truth labels"""
        if self.gan_mode in ["lsgan", "vanilla"]:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == "wgangp":
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


###############################################################################
# CycleGAN Model Wrapper
###############################################################################

class CycleGAN3D:
    """Simplified 3D CycleGAN for image-to-image translation"""
    
    def __init__(self, input_nc=1, output_nc=1, ngf=64, ndf=64, n_blocks=6, 
                 norm="instance", gan_mode="lsgan", init_type="normal", init_gain=0.02):
        """
        Initialize the CycleGAN model
        
        Parameters:
            input_nc (int)  -- number of input channels
            output_nc (int) -- number of output channels
            ngf (int)       -- number of filters in generator
            ndf (int)       -- number of filters in discriminator
            n_blocks (int)  -- number of ResNet blocks
            norm (str)      -- normalization type
            gan_mode (str)  -- GAN loss type
            init_type (str) -- weight initialization type
            init_gain (float) -- scaling factor for initialization
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        norm_layer = get_norm_layer_3d(norm)
        
        # Create generators (use fewer blocks for smaller volumes)
        self.netG_A = ResnetGenerator3D(input_nc, output_nc, ngf, norm_layer=norm_layer, 
                                        use_dropout=False, n_blocks=min(n_blocks, 4))
        self.netG_B = ResnetGenerator3D(output_nc, input_nc, ngf, norm_layer=norm_layer, 
                                        use_dropout=False, n_blocks=min(n_blocks, 4))
        
        # Create discriminators (fewer layers for smaller volumes)
        self.netD_A = NLayerDiscriminator3D(output_nc, ndf, n_layers=2, norm_layer=norm_layer)
        self.netD_B = NLayerDiscriminator3D(input_nc, ndf, n_layers=2, norm_layer=norm_layer)
        
        # Initialize weights
        init_weights(self.netG_A, init_type, init_gain)
        init_weights(self.netG_B, init_type, init_gain)
        init_weights(self.netD_A, init_type, init_gain)
        init_weights(self.netD_B, init_type, init_gain)
        
        # Move to device
        self.netG_A = self.netG_A.to(self.device)
        self.netG_B = self.netG_B.to(self.device)
        self.netD_A = self.netD_A.to(self.device)
        self.netD_B = self.netD_B.to(self.device)
        
        # Define loss functions
        self.criterionGAN = GANLoss(gan_mode).to(self.device)
        self.criterionCycle = nn.L1Loss()
        self.criterionIdt = nn.L1Loss()
        
        # Optimizers
        self.optimizer_G = torch.optim.Adam(
            list(self.netG_A.parameters()) + list(self.netG_B.parameters()),
            lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(
            list(self.netD_A.parameters()) + list(self.netD_B.parameters()),
            lr=0.0002, betas=(0.5, 0.999))
    
    def set_input(self, real_A, real_B):
        """Unpack input data from the dataloader"""
        self.real_A = real_A.to(self.device)
        self.real_B = real_B.to(self.device)
    
    def forward(self):
        """Run forward pass"""
        self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))
        self.fake_A = self.netG_B(self.real_B)  # G_B(B)
        self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B))
    
    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator"""
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D
    
    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)
    
    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)
    
    def backward_G(self, lambda_A=10.0, lambda_B=10.0, lambda_identity=0.5):
        """Calculate the loss for generators G_A and G_B"""
        # Identity loss
        if lambda_identity > 0:
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_identity
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_identity
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0
        
        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
        self.loss_G.backward()
    
    def optimize_parameters(self, lambda_A=10.0, lambda_B=10.0, lambda_identity=0.5):
        """Calculate losses, gradients, and update network weights"""
        # Forward
        self.forward()
        
        # G_A and G_B
        self.optimizer_G.zero_grad()
        self.backward_G(lambda_A, lambda_B, lambda_identity)
        self.optimizer_G.step()
        
        # D_A and D_B
        self.optimizer_D.zero_grad()
        self.backward_D_A()
        self.backward_D_B()
        self.optimizer_D.step()
    
    def get_current_losses(self):
        """Return training losses"""
        return {
            'D_A': self.loss_D_A.item(),
            'G_A': self.loss_G_A.item(),
            'cycle_A': self.loss_cycle_A.item(),
            'idt_A': self.loss_idt_A.item() if isinstance(self.loss_idt_A, torch.Tensor) else self.loss_idt_A,
            'D_B': self.loss_D_B.item(),
            'G_B': self.loss_G_B.item(),
            'cycle_B': self.loss_cycle_B.item(),
            'idt_B': self.loss_idt_B.item() if isinstance(self.loss_idt_B, torch.Tensor) else self.loss_idt_B,
        }
    
    def save_networks(self, save_dir, epoch):
        """Save all networks to disk"""
        import os
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.netG_A.state_dict(), os.path.join(save_dir, f'netG_A_epoch_{epoch}.pth'))
        torch.save(self.netG_B.state_dict(), os.path.join(save_dir, f'netG_B_epoch_{epoch}.pth'))
        torch.save(self.netD_A.state_dict(), os.path.join(save_dir, f'netD_A_epoch_{epoch}.pth'))
        torch.save(self.netD_B.state_dict(), os.path.join(save_dir, f'netD_B_epoch_{epoch}.pth'))
    
    def load_networks(self, save_dir, epoch):
        """Load all networks from disk"""
        import os
        self.netG_A.load_state_dict(torch.load(os.path.join(save_dir, f'netG_A_epoch_{epoch}.pth'), map_location=self.device))
        self.netG_B.load_state_dict(torch.load(os.path.join(save_dir, f'netG_B_epoch_{epoch}.pth'), map_location=self.device))
        self.netD_A.load_state_dict(torch.load(os.path.join(save_dir, f'netD_A_epoch_{epoch}.pth'), map_location=self.device))
        self.netD_B.load_state_dict(torch.load(os.path.join(save_dir, f'netD_B_epoch_{epoch}.pth'), map_location=self.device))
