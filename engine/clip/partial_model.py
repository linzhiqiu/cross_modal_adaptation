from .model import ModifiedResNet, VisionTransformer
import torch
from torch import nn

__all__ = ["get_text_encoder", "get_image_encoder"]


class Encoder(nn.Module):
    def __init__(self, feature_extractor, partial_model):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.partial_model = partial_model


class TransformerEncoder(nn.Module):
    def __init__(self, dtype,
                       token_embedding,
                       positional_embedding=None,
                       transformer_encoder=None,
                       ln_final=None,
                       text_projection=None):
        super().__init__()
        self.dtype = dtype
        self.token_embedding = token_embedding
        self.positional_embedding = positional_embedding
        self.transformer_encoder = transformer_encoder
        self.ln_final = ln_final
        self.text_projection = text_projection
        if self.positional_embedding is None:
            assert self.transformer_encoder is None
        if self.transformer_encoder is None:
            assert self.ln_final is None
        if self.ln_final is None:
            assert self.text_projection is None
    
    def forward(self, text):
        x = self.token_embedding(text).type(self.dtype)  # (bs, seq_len, dim)
        eot_indices = text.argmax(dim=-1)
        if self.positional_embedding is not None:
            x = x + self.positional_embedding.type(self.dtype)

            if self.transformer_encoder is not None:
                x = x.permute(1, 0, 2)  # NLD -> LND
                x = self.transformer_encoder(x)
                x = x.permute(1, 0, 2)  # LND -> NLD

                if self.ln_final is not None:
                    x = self.ln_final(x).type(self.dtype)

                    if self.text_projection is not None:
                        x = x[torch.arange(x.shape[0]), eot_indices] @ self.text_projection
        return x, eot_indices


class PartialTransformer(nn.Module):
    def __init__(self, dtype,
                       logit_scale,
                       vocab_size,
                       positional_embedding=None,
                       partial_transformer=None,
                       ln_final=None,
                       text_projection=None):
        super().__init__()
        self.dtype = dtype
        self.logit_scale = logit_scale
        self.vocab_size = vocab_size
        self.positional_embedding = positional_embedding
        self.partial_transformer = partial_transformer
        self.ln_final = ln_final
        self.text_projection = text_projection
        if self.positional_embedding is not None:
            assert self.partial_transformer is not None
            assert self.ln_final is not None
            assert self.text_projection is not None
        elif self.partial_transformer is not None:
            assert self.ln_final is not None
            assert self.text_projection is not None
        elif self.ln_final is not None:
            assert self.text_projection is not None
    
    def forward(self, x, eot_indices):
        if self.positional_embedding is not None:
            x = x + self.positional_embedding.type(self.dtype)
        
        if self.partial_transformer is not None:
            x = x.permute(1, 0, 2)
            x = self.partial_transformer(x)
            x = x.permute(1, 0, 2)
        
        if self.ln_final is not None:
            x = self.ln_final(x).type(self.dtype)
            x = x[torch.arange(x.shape[0]), eot_indices] @ self.text_projection
        return x



def get_split_transformer(clip_model, layer_idx=0):
    # contains feature_extractor (does encode_text() from prompts) and partial_model (need to reverse the dim)
    vocab_size = clip_model.vocab_size
    token_embedding = clip_model.token_embedding
    positional_embedding = clip_model.positional_embedding
    transformer = clip_model.transformer
    ln_final = clip_model.ln_final
    text_projection = clip_model.text_projection
    logit_scale = clip_model.logit_scale
    dtype = clip_model.dtype

    if layer_idx == -1:
        # finetune all layers
        feature_extractor = TransformerEncoder(
            dtype, token_embedding)
        partial_model = PartialTransformer(
            dtype, logit_scale, vocab_size,
            positional_embedding=positional_embedding,
            partial_transformer=transformer,
            ln_final=ln_final, text_projection=text_projection)
    elif layer_idx == 0:
        # finetune no layers
        feature_extractor = TransformerEncoder(
            dtype, token_embedding,
            positional_embedding=positional_embedding, transformer_encoder=transformer,
            ln_final=ln_final, text_projection=text_projection)
        partial_model = PartialTransformer(dtype, logit_scale, vocab_size)
    else:
        # finetune some layers
        transformer_encoder = transformer.resblocks[:-layer_idx]
        partial_transformer = transformer.resblocks[-layer_idx:]
        feature_extractor = TransformerEncoder(
            dtype, token_embedding,
            positional_embedding=positional_embedding,
            transformer_encoder=transformer_encoder)
        partial_model = PartialTransformer(
            dtype, logit_scale, vocab_size,
            positional_embedding=None,
            partial_transformer=partial_transformer,
            ln_final=ln_final, text_projection=text_projection)
    feature_extractor.eval()
    partial_model.train()
    return Encoder(feature_extractor, partial_model)


def get_text_encoder(text_layer_idx, clip_model):
    # contains feature_extractor (does encode_text() from prompts) and partial_model (need to reverse the dim)
    return get_split_transformer(clip_model, text_layer_idx)


class PartialViT(nn.Module):
    def __init__(self, conv1=None,
                       class_embedding=None,
                       positional_embedding=None,
                       ln_pre=None,
                       transformer_encoder=None,
                       ln_post=None,
                       proj=None,
                 mode='feature_extractor'):
        super().__init__()
        assert mode in ['feature_extractor', 'partial_model']
        self.conv1 = conv1
        self.class_embedding = class_embedding
        self.positional_embedding = positional_embedding
        self.ln_pre = ln_pre
        self.transformer_encoder = transformer_encoder
        self.ln_post = ln_post
        self.proj = proj
        if mode == 'partial_model':
            if self.conv1 is not None:
                assert self.ln_pre is not None
            if self.ln_pre is not None:
                assert self.transformer_encoder is not None
            if self.transformer_encoder is not None:
                assert self.ln_post is not None
            if self.ln_post is not None:
                assert self.proj is not None
        elif mode == 'feature_extractor':
            if self.proj is not None:
                assert self.ln_post is not None
            if self.ln_post is not None:
                assert self.transformer_encoder is not None
            if self.transformer_encoder is not None:
                assert self.ln_pre is not None
            if self.ln_pre is not None:
                assert self.conv1 is not None
    
    def forward(self, x):
        if self.conv1 is not None:
            x = self.conv1(x)
            x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
            x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        
        if self.class_embedding is not None:
            x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        
        if self.positional_embedding is not None:
            x = x + self.positional_embedding.to(x.dtype)
        
        if self.ln_pre is not None:
            x = self.ln_pre(x)
        
        if self.transformer_encoder is not None:
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.transformer_encoder(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
        
        if self.ln_post is not None:
            x = self.ln_post(x[:, 0, :])
        
        if self.proj is not None:
            x = x @ self.proj

        return x


def get_split_vit(model, layer_idx=0):
    # contains feature_extractor and partial_model
    conv1 = model.conv1
    class_embedding = model.class_embedding
    positional_embedding = model.positional_embedding
    ln_pre = model.ln_pre
    transformer = model.transformer
    ln_post = model.ln_post
    proj = model.proj

    if layer_idx == -1:
        # finetune all layers
        feature_extractor = PartialViT(mode='feature_extractor')
        partial_model = PartialViT(conv1=conv1,
                                   class_embedding=class_embedding,
                                   positional_embedding=positional_embedding,
                                   ln_pre=ln_pre,
                                   transformer_encoder=transformer,
                                   ln_post=ln_post,
                                   proj=proj,
                                   mode='partial_model')
    elif layer_idx == 0:
        # finetune no layers
        feature_extractor = PartialViT(conv1=conv1,
                                       class_embedding=class_embedding,
                                       positional_embedding=positional_embedding,
                                       ln_pre=ln_pre,
                                       transformer_encoder=transformer,
                                       ln_post=ln_post,
                                       proj=proj,
                                       mode='feature_extractor')
        partial_model = PartialViT(mode='partial_model')
    else:
        # finetune some layers
        transformer_encoder = transformer.resblocks[:-layer_idx]
        partial_transformer = transformer.resblocks[-layer_idx:]
        feature_extractor = PartialViT(conv1=conv1,
                                       class_embedding=class_embedding,
                                       positional_embedding=positional_embedding,
                                       ln_pre=ln_pre,
                                       transformer_encoder=transformer_encoder,
                                       mode='feature_extractor')
        partial_model = PartialViT(transformer_encoder=partial_transformer,
                                   ln_post=ln_post,
                                   proj=proj,
                                   mode='partial_model')
    feature_extractor.eval()
    partial_model.train()
    return Encoder(feature_extractor, partial_model)


class PartialResNet(nn.Module):
    def __init__(self, conv1=None,
                       bn1=None,
                       conv2=None,
                       bn2=None,
                       conv3=None,
                       bn3=None,
                       layer1=None,
                       layer2=None,
                       layer3=None,
                       layer4=None,
                       attnpool=None,
                       mode='feature_extractor'):
        super().__init__()
        assert mode in ['feature_extractor', 'partial_model']
        self.conv1 = conv1
        self.bn1 = bn1
        self.conv2 = conv2
        self.bn2 = bn2
        self.conv3 = conv3
        self.bn3 = bn3
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)
        self.layer1 = layer1
        self.layer2 = layer2
        self.layer3 = layer3
        self.layer4 = layer4
        self.attnpool = attnpool
        self.apply_stem = self.conv3 != None
        if mode == 'partial_model':
            if self.conv1 is not None:
                assert self.bn1 is not None
            if self.bn1 is not None:
                assert self.conv2 is not None
            if self.conv2 is not None:
                assert self.bn2 is not None
            if self.bn2 is not None:
                assert self.conv3 is not None
            if self.conv3 is not None:
                assert self.conv1 is not None  # make sure entire stem is included
                assert self.bn3 is not None
            if self.bn3 is not None:
                assert self.layer1 is not None
            if self.layer1 is not None:
                assert self.layer2 is not None
            if self.layer2 is not None:
                assert self.layer3 is not None
            if self.layer3 is not None:
                assert self.layer4 is not None
            if self.layer4 is not None:
                assert self.attnpool is not None
        elif mode == 'feature_extractor':
            if self.attnpool is not None:
                assert self.layer4 is not None
            if self.layer4 is not None:
                assert self.layer3 is not None
            if self.layer3 is not None:
                assert self.layer2 is not None
            if self.layer2 is not None:
                assert self.layer1 is not None
            if self.layer1 is not None:
                assert self.bn3 is not None
            if self.bn3 is not None:
                assert self.conv3 is not None
            if self.conv3 is not None:
                assert self.bn2 is not None
            if self.bn2 is not None:
                assert self.conv2 is not None
            if self.conv2 is not None:
                assert self.bn1 is not None
            if self.bn1 is not None:
                assert self.conv1 is not None

    def forward(self, x):
        if self.apply_stem:
            def stem(x):
                for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                    x = self.relu(bn(conv(x)))
                x = self.avgpool(x)
                return x
            x = x.type(self.conv1.weight.dtype)
            x = stem(x)
        if self.layer1 is not None:
            x = self.layer1(x)
        if self.layer2 is not None:
            x = self.layer2(x)
        if self.layer3 is not None:
            x = self.layer3(x)
        if self.layer4 is not None:
            x = self.layer4(x)
        if self.attnpool is not None:
            x = self.attnpool(x)
        return x

def get_split_resnet(model, layer_idx=0):
    # contains feature_extractor and partial_model
    # the 3-layer stem
    conv1 = model.conv1
    bn1 = model.bn1
    conv2 = model.conv2
    bn2 = model.bn2
    conv3 = model.conv3
    bn3 = model.bn3
    avgpool = model.avgpool
    relu = model.relu

    layer1 = model.layer1
    layer2 = model.layer2
    layer3 = model.layer3
    layer4 = model.layer4

    attnpool = model.attnpool

    if layer_idx == -1:
        # finetune all layers
        feature_extractor = PartialResNet(mode='feature_extractor')
        partial_model = PartialResNet(conv1=conv1,
                                      bn1=bn1,
                                      conv2=conv2,
                                      bn2=bn2,
                                      conv3=conv3,
                                      bn3=bn3,
                                      layer1=layer1,
                                      layer2=layer2,
                                      layer3=layer3,
                                      layer4=layer4,
                                      attnpool=attnpool,
                                      mode='partial_model')
    elif layer_idx == 0:
        # finetune no layers
        feature_extractor = PartialResNet(conv1=conv1,
                                          bn1=bn1,
                                          conv2=conv2,
                                          bn2=bn2,
                                          conv3=conv3,
                                          bn3=bn3,
                                          layer1=layer1,
                                          layer2=layer2,
                                          layer3=layer3,
                                          layer4=layer4,
                                          attnpool=attnpool,
                                          mode='feature_extractor')
        partial_model = PartialResNet(mode='partial_model')
    elif layer_idx == 1:
        # finetune attention pool
        feature_extractor = PartialResNet(conv1=conv1,
                                          bn1=bn1,
                                          conv2=conv2,
                                          bn2=bn2,
                                          conv3=conv3,
                                          bn3=bn3,
                                          layer1=layer1,
                                          layer2=layer2,
                                          layer3=layer3,
                                          layer4=layer4,
                                          mode='feature_extractor')
        partial_model = PartialResNet(attnpool=attnpool,
                                      mode='partial_model')
    elif layer_idx == 2:
        # finetune attnpool and layer4
        feature_extractor = PartialResNet(conv1=conv1,
                                          bn1=bn1,
                                          conv2=conv2,
                                          bn2=bn2,
                                          conv3=conv3,
                                          bn3=bn3,
                                          layer1=layer1,
                                          layer2=layer2,
                                          layer3=layer3,
                                          mode='feature_extractor')
        partial_model = PartialResNet(layer4=layer4,
                                      attnpool=attnpool,
                                      mode='partial_model')
    else:
        raise ValueError("Invalid layer index")
    
    feature_extractor.eval()
    partial_model.train()
    return Encoder(feature_extractor, partial_model)


def get_image_encoder(clip_encoder, image_layer_idx, clip_model):
    # contains feature_extractor and partial_model
    if clip_encoder == "RN50":
        assert type(clip_model.visual) == ModifiedResNet
        return get_split_resnet(clip_model.visual, image_layer_idx)
    elif clip_encoder == "ViT-B/16":
        assert type(clip_model.visual) == VisionTransformer
        return get_split_vit(clip_model.visual, image_layer_idx)
    else:
        raise NotImplementedError()