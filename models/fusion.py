import torch
import torch.nn as nn
import torch.nn.functional as F

class NoiseEstimator(nn.Module):
    def __init__(self, in_channels):
        super(NoiseEstimator, self).__init__()
        self.noise_conv1 = nn.Conv2d(in_channels, in_channels//2, kernel_size=3, padding=1)
        self.noise_conv2 = nn.Conv2d(in_channels//2, in_channels//4, kernel_size=3, padding=1)
        self.noise_conv3 = nn.Conv2d(in_channels//4, 1, kernel_size=3, padding=1)
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x):
        local_mean = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        local_var = F.avg_pool2d((x - local_mean)**2, kernel_size=3, stride=1, padding=1)

        noise_feat = self.activation(self.noise_conv1(x))
        noise_feat = self.activation(self.noise_conv2(noise_feat))
        noise_map = torch.sigmoid(self.noise_conv3(noise_feat) + local_var.mean(dim=1, keepdim=True))
        
        return noise_map


class MultiResolutionModule(nn.Module):

    def __init__(self, in_channels):
        super(MultiResolutionModule, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//4, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//4, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//4, kernel_size=5, padding=2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//4, kernel_size=7, padding=3),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.fusion = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        out = torch.cat([branch1, branch2, branch3, branch4], dim=1)
        out = self.fusion(out)
        
        return out


class EnhancedCGA(nn.Module):

    def __init__(self, in_channels, energy_levels=3, reduction_ratio=8):
        super(EnhancedCGA, self).__init__()
        self.energy_levels = energy_levels
        self.channels_per_energy = in_channels // energy_levels

        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, padding_mode='reflect')

        self.energy_pool = nn.AdaptiveAvgPool2d(1)
        self.energy_fc = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.channels_per_energy, self.channels_per_energy // reduction_ratio),
                nn.ReLU(inplace=True),
                nn.Linear(self.channels_per_energy // reduction_ratio, self.channels_per_energy),
                nn.Sigmoid()
            ) for _ in range(energy_levels)
        ])

        self.cross_energy_conv = nn.Conv2d(in_channels, in_channels, 
                                          kernel_size=1, groups=1)
        self.energy_interaction = nn.Conv2d(energy_levels, energy_levels, 
                                           kernel_size=3, padding=1)

        self.pixel_attention = nn.Conv2d(in_channels*3, in_channels, 
                                        kernel_size=7, padding=3, 
                                        padding_mode='reflect', groups=in_channels)

        self.noise_estimator = NoiseEstimator(in_channels)

        self.multi_res = MultiResolutionModule(in_channels)

        self.fusion_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.output_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=1)
        )

    def spatial_attention(self, x):

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention_input = torch.cat([avg_out, max_out], dim=1)
        spatial_map = torch.sigmoid(self.spatial_conv(attention_input))
        return spatial_map
    
    def energy_specific_channel_attention(self, x):
        b, c, h, w = x.size()
        energy_features = []

        for i in range(self.energy_levels):
            start_ch = i * self.channels_per_energy
            end_ch = (i + 1) * self.channels_per_energy
            
            energy_feat = x[:, start_ch:end_ch, :, :]
            energy_pool = self.energy_pool(energy_feat).view(b, -1)
            energy_attn = self.energy_fc[i](energy_pool).view(b, -1, 1, 1)
            energy_feat = energy_feat * energy_attn
            energy_features.append(energy_feat)
        
        return torch.cat(energy_features, dim=1)
    
    def cross_energy_correlation(self, x):
        b, c, h, w = x.size()

        x_reshaped = x.view(b, self.energy_levels, -1, h, w)

        energy_corr = torch.mean(x_reshaped, dim=2)  

        energy_corr = self.energy_interaction(energy_corr)
        
        energy_weights = energy_corr.unsqueeze(2).expand(-1, -1, self.channels_per_energy, -1, -1)
        energy_weights = energy_weights.reshape(b, c, h, w)
        
        enhanced = self.cross_energy_conv(x)
        return enhanced * energy_weights
    
    def forward(self, main_feat, prior_feat):
        batch_size, channels, height, width = main_feat.size()

        initial_feat = main_feat + prior_feat

        spatial_attn = self.spatial_attention(initial_feat)

        channel_attn = self.energy_specific_channel_attention(initial_feat)

        initial_attn = spatial_attn.expand_as(initial_feat) + channel_attn

        cross_energy_feat = self.cross_energy_correlation(initial_feat)
        

        multi_res_feat = self.multi_res(initial_feat)
        

        noise_map = self.noise_estimator(initial_feat)
        

        pixel_input = torch.cat([initial_feat, initial_attn, cross_energy_feat], dim=1)
        pixel_attn = torch.sigmoid(self.pixel_attention(pixel_input))
        

        adaptive_weights = 1.0 - noise_map 
        main_contribution = pixel_attn * main_feat
        prior_contribution = (1.0 - pixel_attn) * prior_feat
        

        fused_feat = initial_feat + (main_contribution * adaptive_weights) + \
                    (prior_contribution * (1.0 - adaptive_weights))
                    

        fused_feat = fused_feat + multi_res_feat

        output = self.fusion_conv(fused_feat)
        output = self.output_conv(output)
        
        return output


class ECGABlock(nn.Module):
    def __init__(self, dim, energy_levels=3):
        super(ECGABlock, self).__init__()
        self.norm1_main = nn.LayerNorm(dim)
        self.norm1_prior = nn.LayerNorm(dim)
        self.ecga = EnhancedCGA(dim, energy_levels)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim)
        )
        
    def forward(self, main_feat, prior_feat):

        main_norm = self.norm1_main(main_feat.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        prior_norm = self.norm1_prior(prior_feat.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        

        attn_out = self.ecga(main_norm, prior_norm)

        x = main_feat + attn_out
        

        x_norm = self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x_shape = x_norm.shape
        x_flat = x_norm.flatten(2).transpose(1, 2)
        x_mlp = self.mlp(x_flat).transpose(1, 2).reshape(x_shape)
        
        return x + x_mlp
