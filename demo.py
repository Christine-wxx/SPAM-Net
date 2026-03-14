import torch
import torch.nn as nn
from models.spam_net import SPAM_Net

def main():
    print("Initializing SPAM-Net (MambaIRv2 + ECGA)...")
    

    upscale = 1 
    

    model = SPAM_Net(
        img_size=512,  
        patch_size=1,
        in_chans=4, 
        embed_dim=48,
        d_state=8,
        depths=(1, 1, 1, 1,),
        num_heads=(1, 1, 1, 1,),
        window_size=16,
        inner_rank=32,
        num_tokens=64,
        convffn_kernel_size=5,
        mlp_ratio=2.,
        qkv_bias=True,
        norm_layer=nn.LayerNorm,
        ape=False,
        patch_norm=True,
        use_checkpoint=False,
        upscale=upscale,       
        img_range=1.,
        upsampler='',   
        resi_connection='1conv',
    ).cuda()



    # Batch = 2, Channels = 4, H = 400, W = 512
    print("\nGenerating dummy test data...")
    _input = torch.randn([2, 4, 400, 512]).cuda()
    

    # Batch = 2, Channels = 1, H = 400, W = 512
    _prior = torch.randn([2, 1, 400, 512]).cuda() 

    print("Running forward pass...")
    with torch.no_grad():
        output = model(_input, _prior)

    print(f"Input shape:  {_input.shape}")
    print(f"Prior shape:  {_prior.shape}")
    print(f"Output shape: {output.shape}")
    
    assert output.shape == _input.shape, "Error: Output shape mismatch!"
    print("\nForward pass successful!")

if __name__ == '__main__':
    main()
