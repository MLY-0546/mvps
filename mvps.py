from typing import Union, List, Optional
import numpy as np
import torch
from pkg_resources import packaging
from torch import nn
from torch.nn import functional as F
from sklearn.cluster import KMeans

from .clip_model import CLIP
from .simple_tokenizer import SimpleTokenizer as _Tokenizer
from .transformer import ResidualAttentionBlock

class ProjectLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_replicas, stack=False, is_array=True):
        super(ProjectLayer, self).__init__()

        self.head = nn.ModuleList([
            nn.Linear(input_dim, output_dim) 
            for _ in range(num_replicas)
        ])
        self.num_replicas = num_replicas
        self.stack = stack
        self.is_array = is_array

    def forward(self, tokens):
        out_tokens = []
        for i in range(self.num_replicas):
            if self.is_array:
                if len(tokens[i].shape) == 3: # For Patch Tokens
                    temp = self.head[i](tokens[i][:, 1:, :]) # for ViT, we exclude the class token and only extract patch tokens here.
                else: # For CLS Tokens
                    temp = self.head[i](tokens[i])
            else:
                if len(tokens.shape) == 3:
                    temp = self.head[i](tokens[:,i,:])
                else :
                    temp = self.head[i](tokens)

            out_tokens.append(temp)

        if self.stack:
            out_tokens = torch.stack(out_tokens, dim=1)

        return out_tokens

class PromptLayer(nn.Module):
    def __init__(self, channel, length, depth, is_text, prompting_type, enabled=True):
        super(PromptLayer, self).__init__()

        self.channel = channel
        self.length = length
        self.depth = depth
        self.is_text = is_text
        self.enabled = enabled

        self.prompting_type = prompting_type

        if self.enabled: # only when enabled, the parameters should be constructed
            if 'S' in prompting_type: # static prompts
                # learnable
                self.static_prompts = nn.ParameterList(
                    [nn.Parameter(torch.empty(self.length, self.channel))
                     for _ in range(self.depth)])

                for single_para in self.static_prompts:
                    nn.init.normal_(single_para, std=0.02)

            if 'D' in prompting_type: # dynamic prompts
                self.dynamic_prompts = [0.] # place holder
            self.scale_prompt = [0.]
            self.prompt_layer_id = 1
            self.prompt_id = 0

    def set_dynamic_prompts(self, dynamic_prompts):
        self.dynamic_prompts = dynamic_prompts

    def set_scale_prompt(self, scale_prompts):
        self.scale_prompt = scale_prompts

    def forward_visual(self, resblock, indx, x, k_x=None, v_x=None, attn_mask: Optional[torch.Tensor] = None, prompt_id : int = 0, prompt_layer_id : int = 0):
        if self.enabled:
            length = self.length

            # only prompt the first J layers
            if indx < self.depth:
#                在最新版本中只保留这个
#                if 'S' in self.prompting_type and 'D' in self.prompting_type: # both
#                     static_prompts = self.static_prompts[indx].unsqueeze(0).expand(x.shape[1], -1, -1)
#                     visual_context = self.dynamic_prompts + static_prompts
#                elif 'S' in self.prompting_type:  # static
#                     static_prompts = self.static_prompts[indx].unsqueeze(0).expand(x.shape[1], -1, -1)
#                     visual_context = static_prompts
#                elif 'D' in self.prompting_type:  # dynamic
#                     visual_context = self.dynamic_prompts
#                else:
#                     print('You should at least choose one type of prompts when the prompting branches are not none.')
#                     raise NotImplementedError
                static_prompts = self.static_prompts[indx].unsqueeze(0).expand(x.shape[1], -1, -1)
                visual_context = static_prompts
                
            if indx == 0:  # for the first layer
                visual_context = visual_context.permute(1, 0, 2).half()
                x = torch.cat([x, visual_context], dim=0)
            else:
                if indx < self.depth:  # replace with learnalbe tokens
                    prefix = x[0:x.shape[0] - length, :, :]
                    visual_context = visual_context.permute(1, 0, 2).half()
                    x = torch.cat([prefix, visual_context], dim=0)
                else:  # keep the same
                    x = x

            if self.prompt_id != prompt_id:
                self.prompt_layer_id = 0
                self.prompt_id = prompt_id

            if indx == 0:
                scale_context = self.scale_prompt[0][0,::].half()
                x = torch.cat([x, scale_context], dim = 0)
            else:
                if self.prompt_layer_id < (self.depth - 1):
                    prefix = x[0:x.shape[0] - length, :, :]
                    scale_context = self.scale_prompt[self.prompt_id][self.prompt_layer_id,::].half()
                    x = torch.cat([prefix, scale_context], dim=0)
                    self.prompt_layer_id += 1
                else:
                    x = x
            
        else:
            x = x

        x, attn_tmp = resblock(q_x=x, k_x=k_x, v_x= v_x, attn_mask=attn_mask)

        if self.enabled:
            tokens = x[0:x.shape[0] - 2 * length, :, :]
            return x, tokens, attn_tmp
        else:
            tokens = x
            return x, tokens, attn_tmp

    def forward(self, resblock, indx, x, k_x=None, v_x=None, attn_mask: Optional[torch.Tensor] = None, prompt_id : int = 0, prompt_layer_id : int = 0):
        if not self.is_text:
            return self.forward_visual(resblock, indx, x, k_x, v_x, attn_mask, prompt_id, prompt_layer_id)

class MutilPrompt_TextEmbebddingLayer(nn.Module):
    def __init__(self, fixed, length):
        super(MutilPrompt_TextEmbebddingLayer, self).__init__()
        self.tokenizer = _Tokenizer()
        self.ensemble_text_features = {}
        self.fixed = fixed
        # 修改为简要的提示
        self.prompt_normal = ['normal {}']
        self.prompt_abnormal = ['abnormal {}' ]
        self.prompt_state = [self.prompt_normal, self.prompt_abnormal]
        self.prompt_templates = ['{}']
        self.prompt_placeholder = " ".join(["X"] * length) 

        self.prompt_normal_length = len(self.prompt_normal) * len(self.prompt_templates)
        self.prompt_abnormal_length = len(self.prompt_abnormal) * len(self.prompt_templates)

    def tokenize(self, texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False, placeholder_setting = True) -> Union[
        torch.IntTensor, torch.LongTensor]:
        if isinstance(texts, str):
            texts = [texts]

        sot_token = self.tokenizer.encoder["<|startoftext|>"]
        eot_token = self.tokenizer.encoder["<|endoftext|>"]
        # 占位符用于后续的软提示替换
        if placeholder_setting:
            all_tokens = [[sot_token] + self.tokenizer.encode(self.prompt_placeholder + ' ' + "object") + [eot_token] for text in texts] 
        else :
            all_tokens = [[sot_token] + self.tokenizer.encode(text) + [eot_token] for text in texts] 
        if packaging.version.parse(torch.__version__) < packaging.version.parse("1.8.0"):
            result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
        else:
            result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

        for i, tokens in enumerate(all_tokens):
            if len(tokens) > context_length:
                if truncate:
                    tokens = tokens[:context_length]
                    tokens[-1] = eot_token
                else:
                    raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
            result[i, :len(tokens)] = torch.tensor(tokens)

        return result

    def encode_text(self, model, texts, device):
        normal_sentence = []
        abnoraml_sentence = []
        for i in range(len(self.prompt_state)):
            for text in texts:
                text = text.replace('_', ' ')
                prompted_state = [state.format(text) for state in self.prompt_state[i]]
                for s in prompted_state:
                    for template in self.prompt_templates:
                        if i == 0:
                            normal_sentence.append(template.format(s))
                        else : 
                            abnoraml_sentence.append(template.format(s))
        
        normal_sentence = self.tokenize(normal_sentence, context_length=77).to(device)
        abnoraml_sentence = self.tokenize(abnoraml_sentence, context_length=77).to(device)
        
        normal_class_embeddings = model.encode_text(normal_sentence, 0)
        abnormal_class_embeddings = model.encode_text(abnoraml_sentence, 1)
        
        class_embeddings = [normal_class_embeddings, abnormal_class_embeddings]
        length = [self.prompt_normal_length, self.prompt_abnormal_length]
        bs = len(texts)

        for state_id in range(len(class_embeddings)):
            for class_embeddings_id in range(len(class_embeddings[state_id])):
                class_embedding = class_embeddings[state_id][class_embeddings_id].reshape(bs, length[state_id], -1)
                class_embedding /= class_embedding.norm(dim=-1, keepdim=True)
                class_embedding = class_embedding.mean(dim=1)
                class_embedding /= class_embedding.norm(dim=-1, keepdim=True)
                class_embeddings[state_id][class_embeddings_id] = class_embedding
    
        if len(self.prompt_state) != 2:
            raise RuntimeError("len(self.prompt_state) must equal to 2")

        text_features = [torch.stack([i, j], dim=1) for i , j in zip(class_embeddings[0], class_embeddings[1])]

        return text_features

    def forward(self, model, texts, device):
        text_features = self.encode_text(model, texts, device)
        for text_feature_id in range(len(text_features)):
            text_features[text_feature_id] = F.normalize(text_features[text_feature_id], dim=-1)
            text_features[text_feature_id] = text_features[text_feature_id].permute(0,2,1)
        return text_features

class MutilPrompt_TextPromptLayer(nn.Module):
    def __init__(self, channel, length, prompt_depth, prompt_num, prompting_type, enabled=True):
        super(MutilPrompt_TextPromptLayer, self).__init__()

        self.channel = channel
        self.length = length

        self.prompt_num = prompt_num
        self.prompt_depth = prompt_depth
        self.enabled = enabled
        self.prompting_type = prompting_type

        if self.enabled:
            self.learnable_prompt = nn.ModuleList()
            for _ in range(2):
                self.learnable_prompt.append(    
                    torch.nn.ParameterList([
                        nn.Parameter(torch.empty(self.length, channel))
                        for _ in range(self.prompt_num)
                    ])            
                )
            
            for learn_para in self.learnable_prompt:
                for single_para in learn_para:
                    nn.init.normal_(single_para, std=0.02)
                        
            if 'S' in prompting_type:
                self.static_prompts = nn.ModuleList()
                for _ in range(self.prompt_num):
                    self.static_prompts.append(
                        nn.ParameterList(
                            [
                                nn.Parameter(torch.empty(self.length, channel))
                                for _ in range(self.prompt_depth - 1)
                            ]                         
                        )    
                    )
                
                for static_prompt in self.static_prompts:
                    for single_para in static_prompt:
                        nn.init.normal_(single_para, std=0.02)

            if 'D' in prompting_type:
                self.dynamic_prompts = [0.]    
            self.frist_dynamic_prompts = None
            self.memory_prompt = [] # 记录提示 

    def set_dynamic_prompts(self, dynamic_prompts):
        self.dynamic_prompts = dynamic_prompts
    
    def set_first_dynamic_prompts(self, dynamic_prompts):
        self.frist_dynamic_prompts = dynamic_prompts

    def reset_memory_prompt(self, ):
        self.memory_prompt = []

    def forward(self, resblock, indx, prompt_id = 0, x = None, k_x=None, v_x=None, attn_mask: Optional[torch.Tensor] = None, state_idx = 0):
        if self.enabled:
            if 0 < indx < self.prompt_depth:
                if 'S' in self.prompting_type and 'D' in self.prompting_type: # both
                    B, tuning_length, C = self.dynamic_prompts[prompt_id].shape
                    _ , BT, C  = x.shape
                    T = BT // B 
                    static_prompts = self.static_prompts[prompt_id][indx-1].unsqueeze(0).expand(x.shape[1], -1, -1)                    
                    dynamic_context = self.dynamic_prompts[prompt_id].unsqueeze(1).repeat_interleave(T, dim=1).view(-1, tuning_length, C)
                    textual_context = dynamic_context + static_prompts
                elif 'S' in self.prompting_type:  # static
                    static_prompts = self.static_prompts[prompt_id][indx-1].unsqueeze(0).expand(x.shape[1], -1, -1)
                    textual_context = static_prompts
                elif 'D' in self.prompting_type:  # dynamic
                    B, tuning_length, C = self.dynamic_prompts[prompt_id].shape
                    _ , BT, C  = x.shape
                    T = BT // B 
                    textual_context = self.dynamic_prompts[prompt_id].unsqueeze(1).repeat_interleave(T, dim=1).view(-1, tuning_length, C)
                else:
                    print('You should at least choose one type of prompts when the prompting branches are not none.')
                    raise NotImplementedError

            if indx == 0:
                prefix = x[:1, :, :]
                suffix = x[1 + self.length:, :, :]

                B, tuning_length, C = self.frist_dynamic_prompts[state_idx][prompt_id].shape
                _ , BT, C  = x.shape
                T = BT // B
                dynamic_context = self.frist_dynamic_prompts[state_idx][prompt_id].unsqueeze(1).repeat_interleave(T, dim=1).view(-1, tuning_length, C)
                dynamic_context = dynamic_context.permute(1,0,2)

                learnable_prompt = self.learnable_prompt[state_idx][prompt_id].unsqueeze(1).repeat(1,prefix.shape[1],1) + dynamic_context
                 
                x = torch.cat([prefix, learnable_prompt, suffix], dim=0)            
            elif indx < self.prompt_depth:
                prefix = x[:1, :, :]
                suffix = x[1 + self.length:, :, :]
                textual_context = textual_context.permute(1, 0, 2)
                x = torch.cat([prefix, textual_context, suffix], dim=0)
                self.memory_prompt.append(textual_context)
            else : 
                x = x
        else:
            x = x

        if resblock is not None:
            x, attn_tmp = resblock(q_x=x, k_x=k_x, v_x= v_x, attn_mask=attn_mask)
            return x, attn_tmp
        else:
            raise RuntimeError("Must input resblock")

class MVPCL(nn.Module):
    def __init__(self, k_clusters, prompt_num, deta_k):
        super(MVPCL, self).__init__()
        self.k_clusters = k_clusters
        self.prompt_num = prompt_num
        
        self.n_aggregate_patch_tokens = [(self.k_clusters - deta_k * prompt_id) * 5 for prompt_id in range(self.prompt_num)]
        self.cluster_performer = [KMeans(n_clusters = self.k_clusters - deta_k * prompt_id, n_init="auto") for prompt_id in range(self.prompt_num)]

        if self.n_aggregate_patch_tokens[-1] <= 0:
            raise RuntimeError("reset these paramter (k_cluseter, group_cnt, deta_k)")

    def forward(self, patch_token: torch.Tensor, anomaly_map: torch.Tensor, prompt_id: int):
        anomaly_map = torch.softmax(anomaly_map, dim=2)[:, :, 1] # B, L

        # extract most abnormal feats
        k = min(anomaly_map.shape[1], self.n_aggregate_patch_tokens[prompt_id])
        top_k_indices = torch.topk(anomaly_map, k=k, dim=1).indices
        
        selected_tokens = patch_token.gather(dim=1, index=top_k_indices.unsqueeze(-1).expand(-1, -1, patch_token.shape[-1]))

        batch_cluster_centers = []

        # use kmeans to extract these centriods
        # Perform K-Means clustering
        for b in range(selected_tokens.shape[0]):
            cluster_labels = self.cluster_performer[prompt_id].fit_predict(selected_tokens[b, :, :].detach().cpu().numpy())

            # Initialize a list to store the cluster centers
            cluster_centers = []

            # Extract cluster centers for each cluster
            for cluster_id in range(self.cluster_performer[prompt_id].n_clusters):
                collected_cluster_data = selected_tokens[b, :, :][cluster_labels == cluster_id]
                cluster_center = torch.mean(collected_cluster_data, dim=0, keepdim=True)
                cluster_centers.append(cluster_center)

            # Normalize the cluster centers
            cluster_centers = torch.cat(cluster_centers, dim=0)
            batch_cluster_centers.append(cluster_centers)

        batch_cluster_centers = torch.stack(batch_cluster_centers, dim=0) # [B, K, D]
        batch_cluster_centers = F.normalize(batch_cluster_centers, dim=-1) # [B, K, D]

        return batch_cluster_centers

class HMPT(nn.Module):
    def __init__(self, text_channel=768, visual_channel=1024, hidden_dim=256):
        super(HMPT, self).__init__()

        self.visual_rca = \
            ResidualAttentionBlock(
                d_model=visual_channel,
                n_head=8,
                is_cross_attention=True,
                n_dropout=.1
            )
        self.scale_adapter = nn.Sequential(
            nn.Linear(visual_channel, hidden_dim),
            nn.Dropout(.1),
            nn.ReLU(),
            nn.Linear(hidden_dim, visual_channel)
        )

    def forward(self, model, patch_tokens, texts, device, patch_token_id):
        gobal_cls_token = patch_tokens[-1][:,0,:]
        if patch_token_id == len(patch_tokens) - 1:
            return [gobal_cls_token, gobal_cls_token]
        
        scale_patch_token = patch_tokens[patch_token_id][:,1:,:]
        gobal_cls_token = self.scale_adapter(gobal_cls_token)
        q =  gobal_cls_token.unsqueeze(1).permute(1,0,2)
        kv = torch.cat([gobal_cls_token.unsqueeze(1), scale_patch_token], dim=1).permute(1,0,2)
        
        gobal_cls_token = \
            self.visual_rca(
                q_x = q, 
                k_x = kv, 
                v_x = kv)[0].permute(1,0,2)[:,0,:]
        return [gobal_cls_token, gobal_cls_token]

class MVPS(nn.Module):
    def __init__(
        self,
        freeze_clip : CLIP,
        text_channel : int , 
        visual_channel: int,
        prompting_length: int,
        prompting_depth: int,
        prompting_branch: str, 
        prompting_type: str,
        use_hsf: bool,
        k_clusters: int,
        output_layers: list,
        device: str,
        image_size: int,
        alpha : float,
    ) -> None:
        super(MVPS, self).__init__()

        # 提示的数量需要和 output_layer 去保持一致
        self.prompt_num = len(output_layers)
        self.output_layers = output_layers
        self.freeze_clip = freeze_clip
        self.visual = self.freeze_clip.visual
        self.transformer = self.freeze_clip.transformer
        self.token_embedding = self.freeze_clip.token_embedding
        self.positional_embedding = self.freeze_clip.positional_embedding
        self.ln_final = self.freeze_clip.ln_final
        self.text_projection = self.freeze_clip.text_projection
        self.attn_mask = self.freeze_clip.attn_mask

        # 设置提示微调参数
        self.prompting_branch = prompting_branch
        self.prompting_type = prompting_type
        self.prompting_depth = prompting_depth
        self.prompting_length = prompting_length
        self.use_hsf = use_hsf
        self.k_clusters = k_clusters

        if 'L' in self.prompting_branch:
            self.enable_text_prompt = True
        else : 
            self.enable_text_prompt = False

        if 'V' in self.prompting_branch:
            self.enable_visual_prompt = True
        else:
            self.enable_visual_prompt = False

        self.text_embedding_layer = MutilPrompt_TextEmbebddingLayer(
            fixed=(not self.enable_text_prompt),
            length=self.prompting_length
        )
        self.text_prompter = MutilPrompt_TextPromptLayer(
            text_channel,
            prompting_length,
            prompt_depth=prompting_depth,
            prompting_type=prompting_type,
            enabled=self.enable_text_prompt,
            prompt_num=self.prompt_num
        )
        self.visual_prompter = PromptLayer(
            visual_channel, 
            prompting_length, 
            prompting_depth, 
            is_text=False,
            prompting_type=prompting_type,
            enabled=self.enable_visual_prompt
        )

        self.patch_token_layer = ProjectLayer(
            visual_channel,
            text_channel,
            len(output_layers), stack=False, is_array=True
        )
        self.cls_token_layer = ProjectLayer(
            text_channel,
            text_channel,
            len(output_layers), stack=False, is_array=True
        )

#        self.dynamic_visual_prompt_generator = \
#            ProjectLayer(
#                visual_channel,
#                visual_channel,
#                prompting_length,
#                stack=True,
#                is_array=False
#            )

        self.DPTO = nn.ModuleList()
        for i in range(self.prompt_num):
            self.DPTO.append(
                nn.Linear(
                    text_channel,
                    visual_channel
                )
            )
        
        self.dynamic_text_prompt_generator = nn.ModuleList()
        for _ in range(3):
            self.dynamic_text_prompt_generator.append(
                ProjectLayer(
                    visual_channel,
                    text_channel,
                    prompting_length,
                    stack=True,
                    is_array=False
                )
            )
        self.mvpcl = MVPCL(prompt_num=self.prompt_num, k_clusters=10, deta_k=0)
        self.hmpt = HMPT(
            text_channel,
            visual_channel,
            256
        )

        self.image_size = image_size
        self.device = device
        self.alpha = 0.5

    def generate_and_set_dynamic_promtps(self, image, texts):
        with torch.no_grad():
            # extract image features
            image_features , patch_tokens = self.visual.forward(image, self.output_layers, single_output=False)

#        if 'D' in self.prompting_type: 
#            visual_prompt = self.hmpt.forward(
#                self,
#                patch_tokens,
#                texts,
#                self.device,
#                patch_token_id=-1
#            )
#            common_visual_prompt = visual_prompt[0] + visual_prompt[1]
#            common_visual_prompt = F.normalize(common_visual_prompt, dim = -1)
#            dynamic_visual_prompts = self.dynamic_visual_prompt_generator(common_visual_prompt)
#            self.visual_prompter.set_dynamic_prompts(dynamic_visual_prompts)

        text_normal_prompt = []
        text_abnormal_prompt = []
        tuning_text_prompt = []
        for patch_token_id in range(len(patch_tokens)):
            text_prompt = self.hmpt.forward(
                self,
                patch_tokens,
                texts,
                self.device,
                patch_token_id=patch_token_id
            )
            text_normal_prompt.append(self.dynamic_text_prompt_generator[0](text_prompt[0]))
            text_abnormal_prompt.append(self.dynamic_text_prompt_generator[1](text_prompt[1]))
            
            common_text_prompt = text_prompt[0] + text_prompt[1]
            common_text_prompt = F.normalize(common_text_prompt, dim = -1)
            tuning_text_prompt.append(
                self.dynamic_text_prompt_generator[-1](
                    common_text_prompt
                )
            )
        
        first_text_promt = [text_normal_prompt, text_abnormal_prompt]

        self.text_prompter.set_first_dynamic_prompts(first_text_promt)
        self.text_prompter.set_dynamic_prompts(tuning_text_prompt)
        
        return image_features, patch_tokens

    def proj_visual_tokens(self, image_features, patch_tokens, ori_visual_feature = None):
        ori_patch_token = ori_visual_feature[1]

        for layer in range(len(patch_tokens)):
            patch_tokens[layer] += ori_patch_token[layer]
            patch_tokens[layer] /= patch_tokens[layer].norm(dim=-1, keepdim=True)
        
        # for patch tokens
        proj_patch_tokens = self.patch_token_layer(patch_tokens)
        for layer in range(len(proj_patch_tokens)):
            proj_patch_tokens[layer] /= proj_patch_tokens[layer].norm(dim=-1, keepdim=True)

        # for cls tokens
        proj_cls_tokens = self.cls_token_layer(image_features)
        for layer in range(len(proj_cls_tokens)):
            proj_cls_tokens[layer] /= proj_cls_tokens[layer].norm(dim=-1, keepdim=True)

        return proj_cls_tokens, proj_patch_tokens
    
    def encode_text(self, text, idx):
        cast_dtype = self.transformer.get_cast_dtype()
        x = self.token_embedding(text).to(cast_dtype)
        x = x + self.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)

        prompt_set = []
        for prompt_id in range(self.prompt_num):
            for indx, r in enumerate(self.transformer.resblocks):
                x, attn_tmp = self.text_prompter(r, indx, prompt_id, x, k_x=None, v_x=None, attn_mask=self.attn_mask, state_idx=idx)
            prompt_set.append(x)
        
        prompt_set = [x.permute(1, 0, 2) for x in prompt_set]  # LND -> NLD
        prompt_set = [self.ln_final(x) for x in prompt_set]  # [batch_size, n_ctx, transformer.width]

        # take features from the eot embedding (eot_token is the highest number in each sequence)
        prompt_set = [x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection for x in prompt_set]
        return prompt_set

    def encode_image(self, image):
        x = image
        # to patches - whether to use dual patchnorm - https://arxiv.org/abs/2302.01327v1
        if self.visual.input_patchnorm:
            # einops - rearrange(x, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)')
            x = x.reshape(x.shape[0], x.shape[1],
                          self.visual.grid_size[0],
                          self.visual.patch_size[0],
                          self.visual.grid_size[1],
                          self.visual.patch_size[1])
            x = x.permute(0, 2, 4, 1, 3, 5)
            x = x.reshape(x.shape[0], self.visual.grid_size[0] * self.visual.grid_size[1], -1)
            x = self.visual.patchnorm_pre_ln(x)
            x = self.visual.conv1(x)
        else:
            x = self.visual.conv1(x)  # shape = [*, width, grid, grid]
            x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
            x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        # class embeddings and positional embeddings
        x = torch.cat(
            [self.visual.class_embedding.to(x.dtype) +
             torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]

        x = x + self.visual.positional_embedding.to(x.dtype)

        # a patch_dropout of 0. would mean it is disabled and this function would do nothing but return what was passed in
        x = self.visual.patch_dropout(x)
        x = self.visual.ln_pre(x)

        patch_embedding = x

        x = x.permute(1, 0, 2)  # NLD -> LND

        patch_tokens = []
        ouput_x_list = []
        prompt_id = 0
        for indx, r in enumerate(self.visual.transformer.resblocks):
            if (indx + 1) in self.output_layers:
                x, tokens, attn_tmp = \
                    self.visual_prompter(r, indx, x, k_x=None, v_x=None, attn_mask=None, prompt_id=prompt_id, )
                patch_tokens.append(tokens)
                ouput_x_list.append(x)
                prompt_id += 1
            else : 
                x, tokens, attn_tmp = \
                    self.visual_prompter(r, indx, x, k_x=None, v_x=None, attn_mask=None, prompt_id=prompt_id)
                
        x = x.permute(1, 0, 2)  # LND -> NLD
        ouput_x_list = [ouput_x_list[t].permute(1, 0, 2) for t in range(len(ouput_x_list))]  # LND -> NLD
        patch_tokens = [patch_tokens[t].permute(1, 0, 2) for t in range(len(patch_tokens))]  # LND -> NLD
        
        img_feats = []
        if self.visual.attn_pool is not None:
            x = self.visual.attn_pool(x)
            x = self.visual.ln_post(x)
            pooled, tokens = self.visual._global_pool(x)
        else:
            for patch_token in patch_tokens:
                pooled, tokens = self.visual._global_pool(patch_token)
                pooled = self.visual.ln_post(pooled)
                img_feats.append(pooled)

        if self.visual.proj is not None:
            img_feats = [pooled @ self.visual.proj for pooled in img_feats]

        return img_feats, patch_tokens, patch_embedding
     
    def visual_text_similarity(self, image_feature, patch_token, text_feature, aggregation, texts):
        anomaly_maps = []
        
        for prompt_id in range(self.prompt_num):  
            anomaly_map = 100.0 * patch_token[prompt_id] @ text_feature[prompt_id]
            anomaly_maps.append(anomaly_map)

        anomaly_scores = []
        
        cls_alpha = 0.5 # self.alpha
        for prompt_id in range(self.prompt_num):
            clustered_feature = self.mvpcl.forward(patch_token[prompt_id], anomaly_maps[prompt_id], prompt_id) # [B,K,D]
            K = clustered_feature.shape[1]
            clustered_feature = cls_alpha * clustered_feature + (1 - cls_alpha) * image_feature[-1].unsqueeze(1).repeat(1,K,1)
            anomaly_score_cluster = 100 * clustered_feature @ text_feature[prompt_id] # [B,K,2]

            anomaly_score_cluster = torch.mean(anomaly_score_cluster, dim=1) # [B,2]
            anomaly_score_cluster = torch.softmax(anomaly_score_cluster, dim=-1) # [B,2]

            anomaly_scores.append(anomaly_score_cluster)

        # NOTE: this bilinear interpolation is not unreproducible and may occasionally lead to unstable ZSAD performance.
        for i in range(len(anomaly_maps)):
            B, L, C = anomaly_maps[i].shape
            H = int(np.sqrt(L))
            anomaly_maps[i] = anomaly_maps[i].permute(0, 2, 1).view(B, 2, H, H)
            anomaly_maps[i] = F.interpolate(anomaly_maps[i], size=self.image_size, mode='bilinear', align_corners=True)

        if aggregation: # in the test stage, we firstly aggregate logits from all hierarchies and then do the softmax normalization
            anomaly_map = torch.mean(torch.stack(anomaly_maps, dim=1), dim=1)
            anomaly_map = torch.softmax(anomaly_map, dim=1)
            anomaly_map = (anomaly_map[:, 1:, :, :] + 1 - anomaly_map[:, 0:1, :, :]) / 2.0
            anomaly_score = torch.mean(torch.stack(anomaly_scores, dim = 1), dim = 1)
            anomaly_score = anomaly_score[:, 1]

            return anomaly_map, anomaly_score
        else: # otherwise, we do the softmax normalization for individual hierarchies
            for i in range(len(anomaly_maps)):
                anomaly_maps[i] = torch.softmax(anomaly_maps[i], dim=1)
            return anomaly_maps, anomaly_scores

    def extract_feat(self, image, cls_name):
        ori_image_features, ori_patch_tokens = self.generate_and_set_dynamic_promtps(image, cls_name) # generate and set dynamic prompts for corresponding prompters
        
        if self.enable_text_prompt:
            text_features = self.text_embedding_layer(self, cls_name, self.device)
        else:
            with torch.no_grad():
                text_features = self.text_embedding_layer(self, cls_name, self.device)

        visual_scale_prompt = []
        text_scale_prompt = self.text_prompter.memory_prompt[:self.prompt_num * (self.prompting_depth - 1)]
        pl , B,  C =  text_scale_prompt[0].shape
        text_scale_prompt = torch.stack(
            text_scale_prompt , 
            dim = 0
        ).reshape(self.prompt_num, self.prompting_depth - 1, pl, B, C) # [12, pl , B, C] -> [4, 3, pl, B, C]
        for prompt_id in range(self.prompt_num):
            visual_scale_prompt.append(
                self.DPTO[prompt_id](
                    text_scale_prompt[prompt_id, ::]
                )
            )
        self.visual_prompter.set_scale_prompt(
            visual_scale_prompt
        )            

        if self.enable_visual_prompt:
            image_features, patch_tokens, _ = self.encode_image(image)
        else:
            with torch.no_grad():
                image_features, patch_tokens, _ = self.encode_image(image)
        
        self.text_prompter.reset_memory_prompt()

        proj_cls_tokens, proj_patch_tokens = self.proj_visual_tokens(image_features, patch_tokens, [ori_image_features, ori_patch_tokens])

        return proj_cls_tokens, proj_patch_tokens, text_features, [ori_image_features, ori_patch_tokens]

    @torch.cuda.amp.autocast()
    def forward(self, image, cls_name, aggregation=True):
        # extract features for images and texts
        image_features, patch_tokens, text_features, ori_visual_feature = self.extract_feat(image, cls_name)
        ori_image_features, ori_patch_tokens = ori_visual_feature
        anomaly_map, anomaly_score = self.visual_text_similarity(image_features, patch_tokens, text_features, aggregation, cls_name)

        if aggregation:
            anomaly_map = anomaly_map # tensor
            anomaly_score = anomaly_score
            anomaly_map = anomaly_map.squeeze(1)

            return anomaly_map, anomaly_score
        else:
            anomaly_maps = anomaly_map # list
            anomaly_score = anomaly_score

            return anomaly_maps, anomaly_score