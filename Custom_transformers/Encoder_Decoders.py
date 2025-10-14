from Custom_transformers.transformeroperations import *
from Custom_transformers.Heads import *
class Encoder:
    def __init__(self,num_heads,d_model):
        self.weights=[]
        self.num_heads=num_heads
        self.norm_layer=AddNormLayer(d_model)
        self.Feed_forward=Feed_Forward(d_model)
        self.mulit_head_attention_layer=Multi_head_attention_layer(d_model,num_heads)
        self.norm_layer2=AddNormLayer(d_model)
        self.weights.extend(self.mulit_head_attention_layer.weights)
        self.weights.extend(self.norm_layer.weights)
        self.weights.extend(self.Feed_forward.weights)
        self.weights.extend(self.norm_layer2.weights)
    def encode(self,source_embedings):
            scores=self.mulit_head_attention_layer.forward(source_embedings)
            norm_scores=self.norm_layer.forward(source_embedings,scores)
            FFn_scores=self.Feed_forward.forward(norm_scores)
            norm_Fnn_socres=self.norm_layer2.forward(norm_scores,FFn_scores)
            return norm_Fnn_socres
    
class Decoder:
    def __init__(self,num_heads,d_model):
        self.weights=[]
        self.num_heads=num_heads
        self.norm_layer=AddNormLayer(d_model)
        self.norm_layer2=AddNormLayer(d_model)
        self.norm_layer3=AddNormLayer(d_model)
        self.Feed_forward=Feed_Forward(d_model)
        self.masked_multi_head_attention_layer=masked_Multi_head_attention_layer(d_model,num_heads)
        self.cross_multi_head_attention_layer=cross_Multi_head_attention_layer(d_model,num_heads)
        self.weights.extend(self.norm_layer3.weights)
        self.weights.extend(self.norm_layer2.weights)
        self.weights.extend(self.norm_layer.weights)
        self.weights.extend(self.masked_multi_head_attention_layer.weights)
        self.weights.extend(self.cross_multi_head_attention_layer.weights)
        self.weights.extend(self.Feed_forward.weights)
        
    def decode(self, encoder_scores, target_embeddings):
        masked_scores = self.masked_multi_head_attention_layer.forward(target_embeddings)
        norm_masked_scores = self.norm_layer.forward(target_embeddings, masked_scores)
        cross_scores = self.cross_multi_head_attention_layer.forward(encoder_scores, norm_masked_scores)
        norm_multi_scores = self.norm_layer2.forward(norm_masked_scores, cross_scores)
        FFN_scores = self.Feed_forward.forward(norm_multi_scores)
        final_norm_scores = self.norm_layer3.forward(norm_multi_scores, FFN_scores)
        return final_norm_scores
    
class DecoderOnly:
    def __init__(self,num_heads,d_model):
        self.weights=[]
        self.num_heads=num_heads
        self.norm_layer=AddNormLayer(d_model)
        self.norm_layer2=AddNormLayer(d_model)
        self.Feed_forward=Feed_Forward(d_model)
        self.masked_multi_head_attention_layer=masked_Multi_head_attention_layer(d_model,num_heads)
        self.weights.extend(self.norm_layer2.weights)
        self.weights.extend(self.norm_layer.weights)
        self.weights.extend(self.masked_multi_head_attention_layer.weights)
        self.weights.extend(self.Feed_forward.weights)
        
    def decode(self, target_embeddings):
        masked_scores = self.masked_multi_head_attention_layer.forward(target_embeddings)
        norm_masked_scores = self.norm_layer.forward(target_embeddings, masked_scores)
        FFN_scores = self.Feed_forward.forward(norm_masked_scores)
        final_norm_scores = self.norm_layer2.forward(norm_masked_scores, FFN_scores)
        return final_norm_scores


class VitEncoder:
    def __init__(self,num_heads,d_model):
        self.weights=[]
        self.num_heads=num_heads
        self.norm_layer=AddNormLayer(d_model)
        self.Feed_forward=Feed_Forward(d_model)
        self.mulit_head_attention_layer=Multi_head_attention_layer(d_model,num_heads)
        self.norm_layer2=AddNormLayer(d_model)
        self.weights.extend(self.mulit_head_attention_layer.weights)
        self.weights.extend(self.norm_layer.weights)
        self.weights.extend(self.Feed_forward.weights)
        self.weights.extend(self.norm_layer2.weights)
    def encode(self,source_embedings):
            scores=self.mulit_head_attention_layer.forward(source_embedings)
            norm_scores=self.norm_layer.forward(source_embedings,scores)
            FFn_scores=self.Feed_forward.forward(norm_scores)
            norm_Fnn_socres=self.norm_layer2.forward(norm_scores,FFn_scores)
            return norm_Fnn_socres
    
class DetrDecoder:
    def __init__(self,num_heads,d_model):
        self.weights=[]
        self.num_heads=num_heads
        self.norm_layer=AddNormLayer(d_model)
        self.norm_layer2=AddNormLayer(d_model)
        self.norm_layer3=AddNormLayer(d_model)
        self.Feed_forward=Feed_Forward(d_model)
        self.masked_multi_head_attention_layer=masked_Multi_head_attention_layer(d_model,num_heads)
        self.cross_multi_head_attention_layer=cross_Multi_head_attention_layer(d_model,num_heads)
        self.weights.extend(self.norm_layer3.weights)
        self.weights.extend(self.norm_layer2.weights)
        self.weights.extend(self.norm_layer.weights)
        self.weights.extend(self.masked_multi_head_attention_layer.weights)
        self.weights.extend(self.cross_multi_head_attention_layer.weights)
        self.weights.extend(self.Feed_forward.weights)
        
    def decode(self, image_featrues, queries):
        masked_scores = self.masked_multi_head_attention_layer.forward(queries)
        norm_masked_scores = self.norm_layer.forward(queries, masked_scores)
        cross_scores = self.cross_multi_head_attention_layer.forward(image_featrues, norm_masked_scores)
        norm_multi_scores = self.norm_layer2.forward(norm_masked_scores, cross_scores)
        FFN_scores = self.Feed_forward.forward(norm_multi_scores)
        final_norm_scores = self.norm_layer3.forward(norm_multi_scores, FFN_scores)
        return final_norm_scores
    