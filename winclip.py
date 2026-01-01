import torch
import open_clip

import torch
import torch.nn.functional as F
import torch.nn as nn

import cv2
from params import MODEL_NAME, DEVICE

class WinCLIP(nn.Module):
    def __init__(self, states, templates, shots=0, option='AC'):
        super().__init__()
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained="winCLIP_ViT_B16P_laion400m.pt")
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(MODEL_NAME)
        self.window_mask = []
        self.visual = self.model.visual
        self.reference_bank = []

        self.state_level = states
        self.template_level = templates
        self.shots = shots
        self.option = option

    @torch.no_grad()
    def encode_image(self, img, windowmask=None, normalize=False):
        """
        WinCLIP image encoder performing both normal image encoding and 
        window-scale encoding by dropping out masked patches before feeding to the CLIP-encoder
        
        :param self: Description
        :param img: query image
        :param windowmask: mask
        :param normalize: normalization for result tokens
        """
        if windowmask is not None:
            x = self.visual.conv1(img)
            x = x.reshape(x.shape[0], x.shape[1], -1)
            x = x.permute(0, 2, 1)

            #concat cls and add positional embedding
            x = torch.cat([self.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
            x = x + self.visual.positional_embedding.to(x.dtype)
            window_cls_list = []
            for mask in windowmask:
                #perform window masking by selecting patches to pass in transformer encoder
                scaled_x = []
                mask = mask.T #now has shape (num_mask, mask_size)
                num_mask, L = mask.shape
                class_index = torch.zeros((mask.shape[0], 1), dtype=torch.int32).to(mask)
                mask = torch.cat((class_index, mask.int()), dim=1)
                for i in mask:
                    mx = torch.index_select(x, 1, i.int())
                    scaled_x.append(torch.index_select(x, 1, i.int()))
                mx = torch.cat(scaled_x)
                mx = self.visual.patch_dropout(mx)
                mx = self.visual.ln_pre(mx)
                mx = self.visual.transformer(mx)

                cls = mx[:, 0]
                cls = self.visual.ln_post(cls)

                if self.visual.proj is not None:
                    cls = cls @ self.visual.proj

                cls /= cls.norm(dim=-1, keepdim=True)

                window_cls_list.append(cls)

            #continue passing into transformer encoder the whole image
            x = self.visual.patch_dropout(x)
            x = self.visual.ln_pre(x)
            x = self.visual.transformer(x)
            cls = x[:, 0]
            tokens = x[:, 1:]
            cls = self.visual.ln_post(cls)
            if self.visual.proj is not None:
                cls = cls @ self.visual.proj

            cls /= cls.norm(dim=-1, keepdim=True)

            tokens /= tokens.norm(dim=-1, keepdim=True)

            return window_cls_list, tokens, cls
        else:
            #getting the image scale feature then just get the cls token
            features = self.model.encode_image(img)
            return features / features.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def encode_text(self, object_name='all'):
        """
        Composition Promt Ensemble generate based on object_name
        
        :param self: Doodoo
        :param object_name: name of the query object
        """
        if object_name[-1].isdigit():
            object_name = object_name[:-1]
        normal_states = [s.format(object_name) for s in self.state_level["normal"]]
        anomaly_states = [s.format(object_name) for s in self.state_level["anomaly"]]

        normal_texts = [t.format(state) for state in normal_states for t in self.template_level]
        anomaly_texts = [t.format(state) for state in anomaly_states for t in self.template_level]

        normal_texts_f = self.tokenizer(normal_texts).to(DEVICE)
        abno_texts_f = self.tokenizer(anomaly_texts).to(DEVICE)
        normal_texts_f = self.model.encode_text(normal_texts_f)
        abno_texts_f = self.model.encode_text(abno_texts_f)

        normal_texts_f = torch.mean(normal_texts_f, dim=0, keepdim=True)
        abno_texts_f = torch.mean(abno_texts_f, dim=0, keepdim=True)

        text_features = torch.cat([normal_texts_f, abno_texts_f], dim=0)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        return text_features

    def gen_window_mask(self, patch_size=16, kernel_size=16, stride=16):
        """generate window scale mask with kernel size"""
        height = 240 // patch_size
        width = 240 // patch_size
        kernel_size = kernel_size // patch_size #calculate window size on patch scale
        stride_size = stride // patch_size
        tmp = torch.arange(1, height*width+1, dtype=torch.float32).reshape(1, 1, height, width)

        mask = torch.nn.functional.unfold(tmp, kernel_size, stride=stride_size)
        return mask
    
    def gen_reference_bank(self, ref_list, masks=None):
        """
        Generates reference banks for few-shot AC/AS, this includes passing
        normal reference images through WinCLIP encoder results in lists of 
        window/image features for each scale

        So based on what's written in the paper, i supposed the reference bank's
        gonna consists of: 
        - For window-scales is gonna be the features which is noted to be the cls tokens
        - For image scale then it's gonna be the patch tokens, since this is used for image 
        guided scoring, the patch score still usefull cause its context is enriched thanks to
        self-attention mechanism
        
        :param self: Doodoo
        :param ref_list: list of reference images
        :param masks: window mask, if none is passed then reference bank only
        includes normal image features
        """
        if masks is None:
            #then expects the window_cls to be empty
            for image in ref_list:
                window_cls, tokens, image_cls = self.encode_image(image, masks)
                #current tokens shape of (B, N, D), since B= 1 then just squeeze it
                token_list = []
                tokens = tokens.squeeze()
                for token in tokens:
                    token_list.append(token)

                self.reference_bank.append(token_list)
        else:
            num_scales = len(masks)
            self.reference_bank = [[] for x in range(num_scales)]
            #one more for image scale
            self.reference_bank.append([])
            for image in ref_list:
                window_cls, tokens, _ = self.encode_image(image, masks)

                for index, cls in enumerate(window_cls):
                    cur_scale = [x for x in cls]
                    self.reference_bank[index].extend(cur_scale)

                token_list = []
                tokens = tokens.squeeze()
                for token in tokens:
                    token_list.append(token)

                self.reference_bank[-1].extend(token_list)

    def calculate_text_anomaly_score(self, text_features, image_features, normal=False):
        """
        Language-guided anomaly scoring function
        Basically perform dot product between normalized tensors for cosine similarity
        
        :param self: Description
        :param text_features: Text features extracted from Composition Promt Ensemble
        :param image_features: Image features :)
        :param normal: wanna get anomaly score or normality score?
        """
        if isinstance(image_features, list):
            score_list = []
            for x in image_features:
                if x.shape[0] != 1:
                    x = x.unsqueeze(0)
                score = (100.0 * x@text_features.T).softmax(dim=-1)

                if not normal:
                    score_list.append(score[:, 1])
                else:
                    score_list.append(score[:, 0])
            return score_list
        else:
            if image_features.shape[0] != 1:
                image_features = image_features.unsqueeze(0)
            score = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            if not normal:
                score = score[:, 1]
            else:
                score = score[:, 0]
            return score
        
    def textual_driven_anomaly_map(self, image_text_score, window_text_score=None, window_list=None, scale_indices=None):
        """
        Generate textual-driven anomaly score map by distributing anomaly score across patches
        (for current model type the map's gonna have shape (255,))
        
        :param image_text_score: ascore0 on image scale
        :param window_text_score: a list containing ascore0 across windows on multiple scale
        :param window_list: list of windows for each scale
        :param scale_indices: starting indices for each scale (used to track window list)
        """
        anomaly_score_map = []
        #distribute window_scale score to each patch
        cur_window_score = torch.zeros(15*15, device=DEVICE)
        cur_patch_weights = torch.zeros(15*15, device=DEVICE)
        for count, (score, window) in enumerate(zip(window_text_score, window_list)):
            #distribute score to a tensor
            #start a new scale then do average on previous scale
            if count in scale_indices:
                cur_window_score = cur_window_score / cur_patch_weights
                anomaly_score_map.append(cur_window_score)

                cur_window_score = torch.zeros(15*15, device=DEVICE)
                cur_patch_weights = torch.zeros(15*15, device=DEVICE)

            window = window.long() - 1
            temp_score = torch.zeros(15 * 15, device=DEVICE)
            temp_weight = torch.zeros(15 * 15, device=DEVICE)
            temp_score[window] = 1.0 / score
            temp_weight[window] = 1.0
            cur_window_score += temp_score
            cur_patch_weights += temp_weight
            count += 1

        #last scale
        cur_window_score = cur_window_score / cur_patch_weights
        anomaly_score_map.append(cur_window_score)

        #distribute image_scale score
        image_scale_score = torch.zeros(15*15, device=DEVICE)
        image_scale_score = torch.full((15*15,), 1.0 / image_text_score.item(), device=DEVICE)
        anomaly_score_map.append(image_scale_score)

        anomaly_score_map = torch.stack(anomaly_score_map, dim=0)
        anomaly_score_map = torch.mean(anomaly_score_map, dim=0)
        anomaly_score_map = 1.0 - 1.0 / anomaly_score_map

        return anomaly_score_map

    def calculate_visual_anomaly_score(self, patch_feature, feature=None, window_masks=None):
        """
        Reference association module for given feature
        return: the visual-guided anomaly score for the query image and the visual-guided anomaly score map

        note: if i'm correct the the visual-guided score for each scale is by taking the highest anomaly score
        from each tokens right since it has the highest possibility for that token to be faulty (cannot find the part
        where they get those Mws or Mwm from)
        
        :param self: Description
        :param feature: a batch of window-scale feature
        :param patch_feature: penultimate feature map (including patch tokens at the end of the encoding phase right before pooling)
        since the cls token of the whole image has gone through a projection layer after returning, it's no more in the
        same dimension with the reference image's patch tokens, so i think using the patch tokens of the query image directly
        might be a better choice
        :param window_masks: window mask (expected a list of generated window for each scale shape (nummask, len))
        """
        # get the list of reference tokens
        anomaly_score_map = []
        if feature is not None:
            #calculate score for each non-image scale features
            for index, (scale, window) in enumerate(zip(feature, window_masks)):
                window = window.T
                cur_reference_bank = self.reference_bank[index]
                cur_reference_bank = torch.stack(cur_reference_bank, dim=0)

                dot_product = (scale@cur_reference_bank.T).max(dim=1)[0]
                scale_score = 0.5 * (1.0 - dot_product)

                per_scale_map = torch.zeros(15 * 15, device=DEVICE)
                per_scale_weight = torch.zeros(15 * 15, device=DEVICE)
                for x, score in zip(window, scale_score):
                    x = x.long() - 1
                    cur_map = torch.zeros(15 * 15, device=DEVICE)
                    cur_weight = torch.zeros(15 * 15, device=DEVICE)
                    cur_map[x] = score
                    cur_weight[x] = 1.0
                    per_scale_map += cur_map
                    per_scale_weight += cur_weight
                
                per_scale_map = per_scale_map / per_scale_weight
                anomaly_score_map.append(per_scale_map)
        

        #calculate association score for image scale feature
        cur_reference_bank = self.reference_bank[-1]
        cur_reference_bank = torch.stack(cur_reference_bank, dim=0)
        patch_feature = F.normalize(patch_feature, dim=-1)        # (num_patches, D)
        cur_reference_bank = F.normalize(cur_reference_bank, dim=-1)  # (num_refs, D)

        patch_feature = patch_feature.squeeze(0)
        dot_product = (patch_feature@cur_reference_bank.T).max(dim=1)[0]
        image_score = 0.5 * (1.0 - dot_product)

        anomaly_score_map.append(image_score.squeeze())
        anomaly_map = torch.stack(anomaly_score_map, dim=0)
        anomaly_map = torch.mean(anomaly_map, dim=0)
        return anomaly_map

    @torch.no_grad()
    def forward(self, object_name, image, ref_list, shot=0, option="AC", out_size=(240, 240)):
        """
        Main pipeline of the model
        
        :param self: Doodoo
        :param object_name: name of the object needed to generate CPE
        :param image: query image (list of query image)
        :param ref_list: list of reference images used for few-shots
        :param shot: number of shot, auto=0 shot
        :param option: 'AC' or 'AS' or None to run all evaluate metrics for both AS and AC (to be updated)
        """
        with torch.no_grad():
            text_features = self.encode_text(object_name)
            if isinstance(image, list):
                image = [x.to(DEVICE) for x in image]
            else:
                image = [image.to(DEVICE)]
            ref_list = [x.to(DEVICE) for x in ref_list]
            if self.option == "AC":
                if self.shots == 0:
                    #WinCLIP zero-shot AC basically applying CPE to CLIP image encoder and text encoder
                    image_features = [self.encode_image(x) for x in image]
                    score = self.calculate_text_anomaly_score(text_features, image_features)
                    
                    score = torch.stack(score)
                    score = torch.mean(score)
                    return score.detach().cpu().numpy()
                else:
                    #adding aggregated vision-based anomaly score
                    scorelist = []
                    window_masks_1 = self.gen_window_mask(kernel_size=32).squeeze().to(DEVICE)
                    window_masks_2 = self.gen_window_mask(kernel_size=48).squeeze().to(DEVICE)
                    window_masks = [window_masks_1, window_masks_2]
                    self.gen_reference_bank(ref_list, window_masks)
                    for x in image:
                        window_cls, tokens, image_cls = self.encode_image(x, window_masks)
                        
                        text_image_score = self.calculate_text_anomaly_score(text_features, image_cls, normal=False)

                        vision_anomaly_map = self.calculate_visual_anomaly_score(tokens, window_cls, window_masks)
                        vision_anomaly_score = vision_anomaly_map.max()

                        AC_score = 0.5 * (text_image_score + vision_anomaly_score)
                        scorelist.append(AC_score)
                    scorelist = torch.stack(scorelist)
                    score = torch.mean(scorelist)
                    return score.detach().cpu().numpy()
            else:
                if self.shots == 0:
                    #zero-shot AS apply multi-scale aggregation score across pixels
                    window_masks_1 = self.gen_window_mask(kernel_size=32).squeeze().to(DEVICE)
                    window_masks_2 = self.gen_window_mask(kernel_size=48).squeeze().to(DEVICE)
                    window_masks = [window_masks_1, window_masks_2]
                    scorelist = []
                    for x in image:
                        window_cls, tokens, image_cls = self.encode_image(x, window_masks)
                    
                        window_cls_list = []
                        for x in window_cls:
                            [window_cls_list.append(window) for window in x]

                        window_text_score = self.calculate_text_anomaly_score(text_features, window_cls_list, normal=True)
                        image_text_score = self.calculate_text_anomaly_score(text_features, image_cls, normal=True)

                        window_list = []
                        scale_indices = []
                        for x in window_masks:
                            x = x.T #(length, num_mask) -> (num_mask, length)
                            index = x.shape[0] if len(scale_indices) == 0 else scale_indices[-1] + x.shape[0]
                            scale_indices.append(index)
                            [window_list.append(window) for window in x]

                        anomaly_score_map = self.textual_driven_anomaly_map(image_text_score, window_text_score=window_text_score, window_list= window_list, scale_indices=scale_indices)
                        scorelist.append(anomaly_score_map)

                    scorelist = torch.stack(scorelist)
                    anomaly_score_map = torch.mean(scorelist, dim=0)

                    anomaly_map = anomaly_score_map.reshape(15, 15).unsqueeze(0)
                    anomaly_map = anomaly_map.unsqueeze(0)

                    anomaly_map = F.interpolate(anomaly_map, size=(240,240), mode='bilinear', align_corners=False)

                    ret_map = anomaly_map.squeeze().detach().cpu().numpy()
                    ret_map = (ret_map - ret_map.min()) / (ret_map.max() - ret_map.min() + 1e-8)
                    return ret_map
                else:
                    #mess with reference images
                    window_masks_1 = self.gen_window_mask(kernel_size=32).squeeze().to(DEVICE)
                    window_masks_2 = self.gen_window_mask(kernel_size=48).squeeze().to(DEVICE)
                    window_masks = [window_masks_1, window_masks_2]
                    self.gen_reference_bank(ref_list, window_masks)

                    maplist = []
                    for x in image:
                        window_cls, tokens, image_cls = self.encode_image(x, window_masks)
                    
                        window_cls_list = []
                        for x in window_cls:
                            [window_cls_list.append(window) for window in x]

                        window_text_score = self.calculate_text_anomaly_score(text_features, window_cls_list, normal=True)
                        image_text_score = self.calculate_text_anomaly_score(text_features, image_cls, normal=True)

                        window_list = []
                        scale_indices = []
                        for x in window_masks:
                            x = x.T #(length, num_mask) -> (num_mask, length)
                            index = x.shape[0] if len(scale_indices) == 0 else scale_indices[-1] + x.shape[0]
                            scale_indices.append(index)
                            [window_list.append(window) for window in x]

                        textual_score_map = self.textual_driven_anomaly_map(image_text_score, window_text_score=window_text_score, window_list= window_list, scale_indices=scale_indices)

                        visual_score_map = self.calculate_visual_anomaly_score(tokens, window_cls, window_masks)

                        # final_map = 1.0 / (1.0 / visual_score_map + 1.0 / anomaly_score_map)
                        final_map = visual_score_map + textual_score_map
                        maplist.append(final_map)

                    final_map = torch.stack(maplist)
                    final_map = torch.mean(final_map, dim=0)
                    final_map = final_map.reshape(15, 15).unsqueeze(0)
                    final_map = final_map.unsqueeze(0)

                    ret_map = F.interpolate(final_map, size=(240,240), mode='bilinear', align_corners=False)

                    ret_map = (ret_map - ret_map.min()) / (ret_map.max() - ret_map.min() + 1e-8)
                    return ret_map.squeeze().detach().cpu().numpy()