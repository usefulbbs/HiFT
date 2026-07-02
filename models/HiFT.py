import torch
import torch.nn as nn
import torch.nn.functional as F
import wespeaker.models.pooling_layers as pooling_layers


class SeparateDownsampling(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(SeparateDownsampling, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        return self.bn(self.conv(x))


class EfficientHiFTLayer(nn.Module):
    def __init__(self, in_channels, bottleneck_channels, segment_length=16):
        super(EfficientHiFTLayer, self).__init__()
        self.segment_length = segment_length
        self.freq_fusion = nn.Sequential(
            nn.Conv1d(in_channels * 2, in_channels, kernel_size=1, groups=in_channels),
            nn.BatchNorm1d(in_channels), 
            nn.ReLU(inplace=True)        
        )
        self.linear1 = nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1)
        self.relu = nn.ReLU()
        self.linear2 = nn.Conv1d(bottleneck_channels, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch, channels, freq, time = x.size()
        
        x_freq_max, _ = torch.max(x, dim=2)   # [B, C, T]
        x_freq_avg = torch.mean(x, dim=2)     # [B, C, T]
        
        x_interleaved = torch.stack([x_freq_max, x_freq_avg], dim=2).flatten(1, 2)
  
        x_freq_pooled = self.freq_fusion(x_interleaved)
       
        global_context = torch.mean(x_freq_pooled, dim=2, keepdim=True) 
        
        seg_len = self.segment_length
        if time % seg_len != 0:
            pad_len = seg_len - (time % seg_len)
            x_padded = F.pad(x_freq_pooled, (0, pad_len))
        else:
            x_padded = x_freq_pooled
            pad_len = 0
            
        num_segments = x_padded.shape[2] // seg_len
        x_segmented = x_padded.contiguous().view(batch, channels, num_segments, seg_len)
        
        segment_context = torch.mean(x_segmented, dim=3, keepdim=True)
        segment_context = segment_context.repeat(1, 1, 1, seg_len)
        segment_context = segment_context.view(batch, channels, -1)
        
        if pad_len > 0:
            segment_context = segment_context[:, :, :-pad_len]
            
        context = global_context + segment_context
        
        mask = self.linear2(self.relu(self.linear1(context)))
        mask = self.sigmoid(mask)
        
        mask = mask.unsqueeze(2) 
        
        return x * mask


class SlimDFResNetBlock_WithHiFT(nn.Module):
    expansion = 2

    def __init__(self, in_planes, planes, stride=1, segment_length=16):
        super(SlimDFResNetBlock_WithHiFT, self).__init__()
        assert stride == 1, "Stride must be 1"
        hidden_planes = planes * self.expansion

        self.conv1 = nn.Conv2d(in_planes, hidden_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_planes)

        self.conv2 = nn.Conv2d(hidden_planes, hidden_planes, kernel_size=3, stride=1, padding=1, groups=hidden_planes, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_planes)

        self.conv3 = nn.Conv2d(hidden_planes, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)

        self.hift = EfficientHiFTLayer(in_channels=planes, 
                                     bottleneck_channels=planes // 4, 
                                     segment_length=segment_length)

        self.shortcut = nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        
        out = self.hift(out)

        out += self.shortcut(x)
        out = F.relu(out)
        return out

class DFResNet155_IntegratedHiFT(nn.Module):
    def __init__(self,
                 num_blocks=[6, 9, 30, 6],
                 m_channels=[32, 64, 128, 256],
                 feat_dim=80,
                 embed_dim=256,
                 pooling_func='TSTP',
                 two_emb_layer=False):
        super(DFResNet155_IntegratedHiFT, self).__init__()
        self.in_planes = m_channels[0]
        self.feat_dim = feat_dim
        self.embed_dim = embed_dim
        self.two_emb_layer = two_emb_layer

        # Stem Layer
        self.conv1 = nn.Conv2d(1, m_channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(m_channels[0])
        self.relu = nn.ReLU(inplace=True)

        # --- Stage 1 (SegLen=4) ---
        self.layer1 = self._make_integrated_layer(
            m_channels[0], num_blocks[0], segment_length=4)
        
        # --- Stage 2 (SegLen=2) ---
        self.ds1 = SeparateDownsampling(m_channels[0], m_channels[1])
        self.layer2 = self._make_integrated_layer(
            m_channels[1], num_blocks[1], segment_length=2)

        # --- Stage 3 (SegLen=1) ---
        self.ds2 = SeparateDownsampling(m_channels[1], m_channels[2])
        self.layer3 = self._make_integrated_layer(
            m_channels[2], num_blocks[2], segment_length=1)

        # --- Stage 4 (SegLen=5) ---
        self.ds3 = SeparateDownsampling(m_channels[2], m_channels[3])
        self.layer4 = self._make_integrated_layer(
            m_channels[3], num_blocks[3], segment_length=5)


        self.final_feat_dim = m_channels[3] * (feat_dim // 8)
        self.pool = getattr(pooling_layers, pooling_func)(in_dim=self.final_feat_dim)
        self.pool_out_dim = self.pool.get_out_dim()

        self.seg_1 = nn.Linear(self.pool_out_dim, embed_dim)
        if self.two_emb_layer:
            self.seg_bn_1 = nn.BatchNorm1d(embed_dim, affine=False)
            self.seg_2 = nn.Linear(embed_dim, embed_dim)
        else:
            self.seg_bn_1 = nn.Identity()
            self.seg_2 = nn.Identity()

    def _make_integrated_layer(self, planes, num_blocks, segment_length):
        layers = []
        for _ in range(num_blocks):

            layers.append(
                SlimDFResNetBlock_WithHiFT(planes, planes, segment_length=segment_length)
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.permute(0, 2, 1).unsqueeze(1)
        out = self.relu(self.bn1(self.conv1(x)))
        
        out = self.layer1(out)
        out = self.ds1(out)
        out = self.layer2(out)
        out = self.ds2(out)
        out = self.layer3(out)
        out = self.ds3(out)
        out = self.layer4(out)

        B, C, F_dim, T = out.shape
        out = out.reshape(B, C * F_dim, T)

        stats = self.pool(out)
        embed_a = self.seg_1(stats)
        
        if self.two_emb_layer:
            out = F.relu(embed_a)
            out = self.seg_bn_1(out)
            embed_b = self.seg_2(out)
            return embed_a, embed_b
        else:
            return torch.tensor(0.0), embed_a


def DFResNet155_FullHiFT(feat_dim, embed_dim, pooling_func='TSTP', two_emb_layer=False):

    return DFResNet155_IntegratedHiFT(
        num_blocks=[6, 9, 30, 6], 
        m_channels=[32, 64, 128, 256],
        feat_dim=feat_dim,
        embed_dim=embed_dim,
        pooling_func=pooling_func,
        two_emb_layer=two_emb_layer
    )

if __name__ == '__main__':
   
    model = DFResNet155_FullHiFT(feat_dim=80, embed_dim=256)
    model.eval()
    
    print("Model initialized successfully.")

    print("\n--- Structural Check (Layer 1) ---")
    print(model.layer1[0]) 
    

    from thop import profile
    x_np = torch.randn(1, 200, 80)
    flops, params = profile(model, inputs=(x_np, ))
    print("FLOPs: {} G, Params: {} M".format(flops / 1e9, params / 1e6))

    x = torch.randn(2, 200, 80)
    _, out = model(x)
    print(f"Input Shape : {x.shape}")
    print(f"Output Shape: {out.shape}")