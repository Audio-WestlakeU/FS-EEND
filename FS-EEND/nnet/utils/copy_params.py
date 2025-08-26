# copy parameters from masked embedding encoder, conv1d, and masked attractor decoder 
# to streaming transformer encoder, streaming conv1d, and streaming attractor decoder

import torch
import torch.nn as nn

def copy_params_with_masked_emb_encoder(masked_enc, streaming_enc):
    
    streaming_enc.bn.load_state_dict(masked_enc.bn.state_dict())
    streaming_enc.proj.load_state_dict(masked_enc.encoder.state_dict())
    streaming_enc.proj_norm.load_state_dict(masked_enc.encoder_norm.state_dict())
    
    for std_layer, incr_layer in zip(masked_enc.transformer_encoder.layers, streaming_enc.layers):
        incr_layer.self_attn.attention.in_proj_weight.data.copy_(std_layer.self_attn.in_proj_weight.data)
        incr_layer.self_attn.attention.in_proj_bias.data.copy_(std_layer.self_attn.in_proj_bias.data)
        incr_layer.self_attn.attention.out_proj.weight.data.copy_(std_layer.self_attn.out_proj.weight.data)
        incr_layer.self_attn.attention.out_proj.bias.data.copy_(std_layer.self_attn.out_proj.bias.data)

        incr_layer.linear1.weight.data.copy_(std_layer.linear1.weight.data)
        incr_layer.linear1.bias.data.copy_(std_layer.linear1.bias.data)
        incr_layer.linear2.weight.data.copy_(std_layer.linear2.weight.data)
        incr_layer.linear2.bias.data.copy_(std_layer.linear2.bias.data)

        incr_layer.norm1.weight.data.copy_(std_layer.norm1.weight.data)
        incr_layer.norm1.bias.data.copy_(std_layer.norm1.bias.data)
        incr_layer.norm2.weight.data.copy_(std_layer.norm2.weight.data)
        incr_layer.norm2.bias.data.copy_(std_layer.norm2.bias.data)

def copy_params_with_conv1d(standard_conv1d: nn.Module, streaming_conv1d: nn.Module):
    streaming_conv1d.conv.load_state_dict(standard_conv1d.state_dict())

def copy_params_with_masked_decoder(masked_dec, streaming_dec):
    streaming_dec.pos_enc.load_state_dict(masked_dec.pos_enc.state_dict())
    streaming_dec.convert.load_state_dict(masked_dec.convert.state_dict())
    
    for msk_layer, str_layer in zip(masked_dec.attractor_decoder.layers, streaming_dec.layers):
        str_layer.temp_attn.attention.in_proj_weight.data.copy_(msk_layer.self_attn1.in_proj_weight.data)
        str_layer.temp_attn.attention.in_proj_bias.data.copy_(msk_layer.self_attn1.in_proj_bias.data)
        str_layer.temp_attn.attention.out_proj.weight.data.copy_(msk_layer.self_attn1.out_proj.weight.data)
        str_layer.temp_attn.attention.out_proj.bias.data.copy_(msk_layer.self_attn1.out_proj.bias.data)
            
        str_layer.spk_attn.in_proj_weight.data.copy_(msk_layer.self_attn2.in_proj_weight.data)
        str_layer.spk_attn.in_proj_bias.data.copy_(msk_layer.self_attn2.in_proj_bias.data)
        str_layer.spk_attn.out_proj.weight.data.copy_(msk_layer.self_attn2.out_proj.weight.data)
        str_layer.spk_attn.out_proj.bias.data.copy_(msk_layer.self_attn2.out_proj.bias.data)
            
        str_layer.linear1.weight.data.copy_(msk_layer.linear1.weight.data)
        str_layer.linear1.bias.data.copy_(msk_layer.linear1.bias.data)
        str_layer.linear2.weight.data.copy_(msk_layer.linear2.weight.data)
        str_layer.linear2.bias.data.copy_(msk_layer.linear2.bias.data)
            
        str_layer.norm1.weight.data.copy_(msk_layer.norm11.weight.data)
        str_layer.norm1.bias.data.copy_(msk_layer.norm11.bias.data)
        str_layer.norm2.weight.data.copy_(msk_layer.norm21.weight.data)
        str_layer.norm2.bias.data.copy_(msk_layer.norm21.bias.data)
        str_layer.norm3.weight.data.copy_(msk_layer.norm22.weight.data)
        str_layer.norm3.bias.data.copy_(msk_layer.norm22.bias.data)

def copy_params_from_masked_to_streaming(masked_fs_eend: nn.Module, streaming_fs_eend: nn.Module):
    copy_params_with_masked_emb_encoder(masked_fs_eend.enc, streaming_fs_eend.enc)
    copy_params_with_conv1d(masked_fs_eend.cnn, streaming_fs_eend.cnn)
    copy_params_with_masked_decoder(masked_fs_eend.dec, streaming_fs_eend.dec)
    