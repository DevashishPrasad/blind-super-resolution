import torch
from torch import nn
from torchsummary import summary

n_feat = 128
kernel_size = 3

class _Res_Block(nn.Module):
    def __init__(self):
        super(_Res_Block, self).__init__()

        self.res_conv = nn.Conv2d(n_feat, n_feat, kernel_size, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.relu(self.res_conv(x))
        y = self.res_conv(y)
        y *= 0.1
        y = torch.add(y, x)
        return y


class edsr(nn.Module):
    def __init__(self, scale=4):
        super(edsr, self).__init__()

        in_ch = 3
        num_blocks = 10

        self.conv1 = nn.Conv2d(in_ch, n_feat, kernel_size, padding=1)
        self.conv_up = nn.Conv2d(n_feat, n_feat * (4 ** (scale//2)), kernel_size, padding=1)
        self.conv_out = nn.Conv2d(n_feat, in_ch, kernel_size, padding=1)

        self.body = self.make_layer(_Res_Block, num_blocks)

        self.upsample = nn.Sequential(self.conv_up, nn.PixelShuffle(scale))

    def make_layer(self, block, layers):
        res_block = []
        for _ in range(layers):
            res_block.append(block())
        return nn.Sequential(*res_block)

    def forward(self, x):
        out = self.conv1(x)
        out = self.body(out)
        out = self.upsample(out)
        out = self.conv_out(out)

        return out

if __name__ == "__main__":
    # Memory profiling
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r-a  # free inside reserved
    print("Total memory:", t)
    print("Reserved memory:", r)
    print("Allocated memory:", a)
    print("Free memory:", f)

    # Model
    model = edsr().cuda()
    mem = torch.cuda.memory_allocated()
    print("Memory allocated: {}".format(mem/1024/1024))
    
    # Summary
    # summary(model, (3, 48, 48))
    print('\n\nNumber of trainable model parameters: {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    # Test
    input = torch.randn(32, 3, 48, 48).cuda()
    output = model(input)
    print(f"Output shape = {output.shape}")

    # # visualize
    # from torchviz import make_dot

    # make_dot(output, params=dict(list(model.named_parameters()))).render("edsr_torchviz", format="png")