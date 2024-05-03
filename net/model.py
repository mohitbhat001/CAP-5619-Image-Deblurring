import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
from einops import rearrange
from bitnet import BitNetTransformer, BitFeedForward



##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)



##########################################################################
## Channel-Wise Cross Attention (CA)
class Chanel_Cross_Attention(nn.Module):
    def __init__(self, dim, num_head):
        super(Chanel_Cross_Attention, self).__init__()
        self.num_head = num_head
        self.temperature = nn.Parameter(torch.ones(num_head, 1, 1), requires_grad=True)

        self.q = nn.Conv2d(dim, dim, kernel_size=1)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)


        self.kv = nn.Conv2d(dim, dim*2, kernel_size=1)
        self.kv_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x, y):
        # x -> q, y -> kv
        assert x.shape == y.shape, 'The shape of feature maps from image and features are not equal!'

        b, c, h, w = x.shape

        q = self.q_dwconv(self.q(x))
        kv = self.kv_dwconv(self.kv(y))
        k, v = kv.chunk(2, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_head)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_head)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_head)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = q @ k.transpose(-2, -1) * self.temperature
        attn = attn.softmax(dim=-1)

        out = attn @ v

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_head, h=h, w=w)

        out = self.project_out(out)
        return out


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.proj(x)
        return x
    

##########################################################################
## H-L Unit
class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()

        self.spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3)

    def forward(self, x):
        max = torch.max(x,1,keepdim=True)[0]
        mean = torch.mean(x,1,keepdim=True)
        scale = torch.cat((max, mean), dim=1)
        scale =self.spatial(scale)
        scale = F.sigmoid(scale)
        return scale

##########################################################################
## L-H Unit
class ChannelGate(nn.Module):
    def __init__(self, dim):
        super(ChannelGate, self).__init__()
        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.max = nn.AdaptiveMaxPool2d((1,1))

        self.mlp = nn.Sequential(
            nn.Conv2d(dim, dim//16, 1),
            nn.ReLU(),
            nn.Conv2d(dim//16, dim, 1)
        )

    def forward(self, x):
        avg = self.mlp(self.avg(x))
        max = self.mlp(self.max(x))

        scale = avg + max
        scale = F.sigmoid(scale)
        return scale

##########################################################################
## Frequency Modulation Module (FMoM)
class FreRefine(nn.Module):
    def __init__(self, dim):
        super(FreRefine, self).__init__()

        self.SpatialGate = SpatialGate()
        self.ChannelGate = ChannelGate(dim)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, low, high):
        spatial_weight = self.SpatialGate(high)
        channel_weight = self.ChannelGate(low)
        high = high * channel_weight
        low = low * spatial_weight

        out = low + high
        out = self.proj(out)
        return out
    
##########################################################################
## Adaptive Frequency Learning Block (AFLB)
class FreModule(nn.Module):
    def __init__(self, dim, heads, in_dim=3):
        super(FreModule, self).__init__()

        self.conv = nn.Conv2d(in_dim, dim, kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Conv2d(in_dim, dim, kernel_size=3, stride=1, padding=1)

        self.score_gen = nn.Conv2d(2, 2, 7, padding=3)

        self.para1 = nn.Parameter(torch.zeros(dim, 1, 1))
        self.para2 = nn.Parameter(torch.ones(dim, 1, 1))

        self.channel_cross_l = Chanel_Cross_Attention(dim, num_head=heads)
        self.channel_cross_h = Chanel_Cross_Attention(dim, num_head=heads)
        self.channel_cross_agg = Chanel_Cross_Attention(dim, num_head=heads)

        self.frequency_refine = FreRefine(dim)

        self.rate_conv = nn.Sequential(
            nn.Conv2d(dim, dim//8, 1),
            nn.GELU(),
            nn.Conv2d(dim//8, 2, 1),
        )

    def forward(self, x, y):
        _, _, H, W = y.size()
        x = F.interpolate(x, (H,W), mode='bilinear')
        
        high_feature, low_feature = self.fft(x) 

        high_feature = self.channel_cross_l(high_feature, y)
        low_feature = self.channel_cross_h(low_feature, y)

        agg = self.frequency_refine(low_feature, high_feature)
        out = self.channel_cross_agg(y, agg)

        return out * self.para1 + y * self.para2

    def shift(self, x):
        '''shift FFT feature map to center'''
        b, c, h, w = x.shape
        return torch.roll(x, shifts=(int(h/2), int(w/2)), dims=(2,3))

    def unshift(self, x):
        """converse to shift operation"""
        b, c, h ,w = x.shape
        return torch.roll(x, shifts=(-int(h/2), -int(w/2)), dims=(2,3))

    def fft(self, x, n=128):
        """obtain high/low-frequency features from input"""
        x = self.conv1(x)
        mask = torch.zeros(x.shape).to(x.device)
        h, w = x.shape[-2:]
        threshold = F.adaptive_avg_pool2d(x, 1)
        threshold = self.rate_conv(threshold).sigmoid()

        for i in range(mask.shape[0]):
            h_ = (h//n * threshold[i,0,:,:]).int()
            w_ = (w//n * threshold[i,1,:,:]).int()

            mask[i, :, h//2-h_:h//2+h_, w//2-w_:w//2+w_] = 1

        fft = torch.fft.fft2(x, norm='forward', dim=(-2,-1))
        fft = self.shift(fft)
        
        fft_high = fft * (1 - mask)

        high = self.unshift(fft_high)
        high = torch.fft.ifft2(high, norm='forward', dim=(-2,-1))
        high = torch.abs(high)

        fft_low = fft * mask

        low = self.unshift(fft_low)
        low = torch.fft.ifft2(low, norm='forward', dim=(-2,-1))
        low = torch.abs(low)

        return high, low


##########################################################################
# Lightweight Gated Feature Fusion Module (LGFF)
class LGFF(nn.Module):
    def __init__(self, in_dim, out_dim, ff_mult):
        super(LGFF, self).__init__()
        self.project_in = nn.Sequential(nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1, groups=in_dim),
                                          nn.Conv2d(in_dim, out_dim, kernel_size=1))
        self.norm = nn.LayerNorm(out_dim)
        self.ffn = BitFeedForward(out_dim, ff_mult)
        
    def forward(self, x):
        x = self.project_in(x)
        x = x + self.ffn(self.norm(x))
        return x
        

##########################################################################
##---------- DeBlurModel -----------------------

class DeBlurModel(nn.Module):
    def __init__(self, 
        inp_channels=3, 
        out_channels=3, 
        dim = 48,
        num_blocks = [4,6,6,8], 
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ff_mult = 2.66,
        decoder = True,
    ):

        super(DeBlurModel, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)        
        self.decoder = decoder
        
        if self.decoder:
            self.fre1 = FreModule(dim*2**3, heads=heads[2])
            self.fre2 = FreModule(dim*2**2, heads=heads[2])
            self.fre3 = FreModule(dim*2**1, heads=heads[2])            

        self.encoder_level1 = nn.Sequential(*[BitNetTransformer(num_tokens = dim, dim=dim, heads=heads[0], ff_mult=ff_mult, depth=1) for i in range(num_blocks[0])])
        
        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2

        self.encoder_level2 = nn.Sequential(*[BitNetTransformer(num_tokens = int(dim*2**1), dim=int(dim*2**1), heads=heads[1], ff_mult=ff_mult, depth=1) for i in range(num_blocks[1])])
        
        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3

        self.encoder_level3 = nn.Sequential(*[BitNetTransformer(num_tokens =int(dim*2**2), dim=int(dim*2**2), heads=heads[2], ff_mult=ff_mult, depth=1) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[BitNetTransformer(num_tokens=int(dim*2**3),dim=int(dim*2**3), heads=heads[3], ff_mult=ff_mult, depth=1) for i in range(num_blocks[3])])
        
        self.up4_3 = Upsample(int(dim*2**3)) ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1)

        self.decoder_level3 = nn.Sequential(*[BitNetTransformer(num_tokens=int(dim*2**2), dim=int(dim*2**2), heads=heads[2], ff_mult=ff_mult, depth=1) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim*2**2)) 
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1)
        self.decoder_level2 = nn.Sequential(*[BitNetTransformer(num_tokens=int(dim*2**1), dim=int(dim*2**1), heads=heads[1], ff_mult=ff_mult, depth=1) for i in range(num_blocks[1])])
        
        self.up2_1 = Upsample(int(dim*2**1)) 

        #self.decoder_level1 = nn.Sequential(*[BitNetTransformer(dim=int(dim*2**1), heads=heads[0], ff_mult=ff_mult, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.decoder_level1 = nn.Sequential(*[BitNetTransformer(num_tokens=int(dim), dim=int(dim), heads=heads[0], ff=ff_mult, depth=1) for i in range(num_blocks[0])])
        
        #self.refinement = nn.Sequential(*[BitNetTransformer(dim=int(dim*2**1), heads=heads[0], ff_mult=ff_mult, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
        self.refinement = nn.Sequential(*[BitNetTransformer(num_tokens=int(dim), dim=int(dim), heads=heads[0], ff_mult=ff_mult, depth=1) for i in range(num_refinement_blocks)])
                    
        #self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.output = nn.Conv2d(int(dim), out_channels, kernel_size=3, stride=1, padding=1)

        # Introducing LLGF Blocks
        self.llgf1 = LGFF(dim*2**1, dim, ff_mult)  # After Level 1
        self.llgf2 = LGFF(dim*2**2, dim*2**1, ff_mult)  # After Level 2
        self.llgf3 = LGFF(dim*2**3, dim*2**2, ff_mult)  # After Level 3

        self.llgf_enc_level1 = LGFF(720, dim, ff_mult)
        self.llgf_enc_level2 = LGFF(720, dim*2, ff_mult)
        self.llgf_enc_level3 = LGFF(720, dim*4, ff_mult)


    def forward(self, inp_img,noise_emb = None):

        inp_enc_level1 = self.patch_embed(inp_img)
        #print(f"input_enc_level1 {inp_enc_level1.shape}")

        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        #print(f"out_enc_level1 {out_enc_level1.shape}")
        
        inp_enc_level2 = self.down1_2(out_enc_level1)
        #print(f"inp_enc_level2 {inp_enc_level2.shape}")

        out_enc_level2 = self.encoder_level2(inp_enc_level2)
        #print(f"out_enc_level2 {out_enc_level2.shape}")

        inp_enc_level3 = self.down2_3(out_enc_level2)
        #print(f"inp_enc_level3 {inp_enc_level3.shape}")

        out_enc_level3 = self.encoder_level3(inp_enc_level3)
        #print(f"out_enc_level3 {out_enc_level3.shape}")

        inp_enc_level4 = self.down3_4(out_enc_level3)    
        #print(f"inp_enc_level4 {inp_enc_level4.shape}")
        
        latent = self.latent(inp_enc_level4) 
        #print(f"latent {latent.shape}")

        if self.decoder:
            latent = self.fre1(inp_img, latent)
            #print(f"latent decoder {latent.shape}")

        out_enc_level1_2 = F.interpolate(out_enc_level1, scale_factor=0.5)
        out_enc_level1_3 = F.interpolate(out_enc_level1, scale_factor=0.25)

        out_enc_level2_1 = F.interpolate(out_enc_level2, scale_factor=2)
        out_enc_level2_3 = F.interpolate(out_enc_level2, scale_factor=0.5)

        out_enc_level3_1 = F.interpolate(out_enc_level3, scale_factor=4)
        out_enc_level3_2 = F.interpolate(out_enc_level3, scale_factor=2)

        latent_3 = F.interpolate(latent, scale_factor=2)
        latent_2 = F.interpolate(latent_3, scale_factor=2)
        latent_1 = F.interpolate(latent_2, scale_factor=2)
       
        out_enc_level1 = self.llgf_enc_level1(torch.cat([out_enc_level3_1, out_enc_level2_1, out_enc_level1, latent_1], dim=1))
        #print(out_enc_level1.shape)
        out_enc_level2 = self.llgf_enc_level2(torch.cat([out_enc_level3_2, out_enc_level2, out_enc_level1_2, latent_2], dim=1))
        #print(out_enc_level2.shape)
        out_enc_level3 = self.llgf_enc_level3(torch.cat([out_enc_level3, out_enc_level2_3, out_enc_level1_3, latent_3], dim=1))
        #print(out_enc_level3.shape)
      
        inp_dec_level3 = self.up4_3(latent)
        #print(f"inp_dec_level3 {inp_dec_level3.shape}")

        inp_dec_level3 = self.llgf3(torch.cat([inp_dec_level3, out_enc_level3], dim=1))
        #inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], dim=1)
        #print(f"inp_dec_level3 {inp_dec_level3.shape}")
        #inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        #print(f"inp_dec_level3 {inp_dec_level3.shape}")

        out_dec_level3 = self.decoder_level3(inp_dec_level3)
        #print(f"out_dec_level3 {out_dec_level3.shape}")

        if self.decoder:
            out_dec_level3 = self.fre2(inp_img, out_dec_level3)
            #print(f"out_dec_level3 decoder {out_dec_level3.shape}")

        inp_dec_level2 = self.up3_2(out_dec_level3)
        #print(f"inp_dec_level2 {inp_dec_level2.shape}")
        inp_dec_level2 = self.llgf2(torch.cat([inp_dec_level2, out_enc_level2], 1))
        #print(f"inp_dec_level2 {inp_dec_level2.shape}")
        #inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        #print(f"inp_dec_level2 {inp_dec_level2.shape}")


        out_dec_level2 = self.decoder_level2(inp_dec_level2)
        #print(f"out_dec_level2 {out_dec_level2.shape}")

        if self.decoder:
            out_dec_level2 = self.fre3(inp_img, out_dec_level2)
            #print(f"out_dec_level2 decoder {out_dec_level2.shape}")

        inp_dec_level1 = self.up2_1(out_dec_level2)
        #print(f"inp_dec_level1 {inp_dec_level1.shape}")
        inp_dec_level1 = self.llgf1(torch.cat([inp_dec_level1, out_enc_level1], 1))
        #print(f"inp_dec_level1 {inp_dec_level1.shape}")
        
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        #print(f"out_dec_level1 {out_dec_level1.shape}")

        out_dec_level1 = self.refinement(out_dec_level1)
        #print(f"out_dec_level1 {out_dec_level1.shape}")

        out_dec_level1 = self.output(out_dec_level1) + inp_img
        #print(f"out_dec_level1 {out_dec_level1.shape}")
        return out_dec_level1