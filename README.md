# SGCAM-
This code implements segmentation and then scene classification for remote sensing images
Implementation Steps  
Step 1: Preprocess the Images  
def preprocess_image(image): 
    # 1. CLAHE Enhancement on luminance channel 
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB) 
    l, a, b = cv2.split(lab) 
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) 
    l_enhanced = clahe.apply(l) 
    enhanced_lab = cv2.merge([l_enhanced, a, b]) 
    enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB) 
     
    # 2. Resize to 512x512 
    resized = cv2.resize(enhanced_rgb, (512, 512)) 
     
    # 3. ImageNet Normalization 
    mean = [0.485, 0.456, 0.406] 
    std = [0.229, 0.224, 0.225] 
    normalized = (resized - mean) / std 
     
    return normalized 
 
Step 2: Semantic Stream (SegFormer) 
from transformers import SegformerForSemanticSegmentation 
import torch.nn.functional as F 
 
class SemanticFeatureEncoder(nn.Module): 
    def __init__(self, num_classes=19): 
        super().__init__() 
        # SegFormer-B2 pretrained on ADE20K 
        self.segformer = SegformerForSemanticSegmentation.from_pretrained( 
            "nvidia/segformer-b2-finetuned-ade-512-512" 
        ) 
        self.num_classes = num_classes 
         
    def forward(self, x): 
        # Get encoder features at multiple scales 
        encoder_outputs = self.segformer.segformer.encoder(x) 
         
        # Fuse multi-scale features 
        fused_features = [] 
        for i, feat in enumerate(encoder_outputs): 
            # Project to common dimension 
            proj_feat = self.projection_layers[i](feat) 
            # Upsample to highest resolution 
            up_feat = F.interpolate(proj_feat,  
                                  size=encoder_outputs[0].shape[-2:], 
                                  mode='bilinear', 
                                  align_corners=False) 
            fused_features.append(up_feat) 
         
        # Element-wise addition of all features 
        semantic_features = torch.sum(torch.stack(fused_features), dim=0) 
         
        # Generate segmentation logits 
        logits = self.segformer.decode_head(semantic_features) 
         
        return semantic_features, logits 
 
Step 3: Segmentation-Guided Cross Attention Module (SG-CAM) 
class SegmentationGuidedCAM(nn.Module): 
    def __init__(self, visual_dim=1280, semantic_dim=256, hidden_dim=256): 
        super().__init__() 
         
        # Projection layers 
        self.q_proj = nn.Linear(visual_dim, hidden_dim) 
        self.k_proj = nn.Linear(semantic_dim, hidden_dim) 
        self.v_proj = nn.Linear(semantic_dim, hidden_dim) 
         
        # Output projection 
        self.out_proj = nn.Linear(hidden_dim, visual_dim) 
         
        # Layer normalization 
        self.norm = nn.LayerNorm(visual_dim) 
         
    def forward(self, visual_features, semantic_features): 
        """ 
        Args: 
            visual_features: [B, C_v, H, W] from CNN 
            semantic_features: [B, C_s, H, W] from SegFormer 
        """ 
        B, C_v, H_v, W_v = visual_features.shape 
        _, C_s, H_s, W_s = semantic_features.shape 
         
        # Reshape to token sequences 
        visual_tokens = visual_features.view(B, C_v, -1).permute(0, 2, 1)  # [B, N_v, C_v] 
        semantic_tokens = semantic_features.view(B, C_s, -1).permute(0, 2, 1)  # [B, 
N_s, C_s] 
         
        # Project to query, key, value spaces 
        Q = self.q_proj(visual_tokens)  # [B, N_v, D] 
        K = self.k_proj(semantic_tokens)  # [B, N_s, D] 
        V = self.v_proj(semantic_tokens)  # [B, N_s, D] 
         
        # Compute cross-attention 
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.hidden_dim ** 
0.5) 
        attention_weights = F.softmax(attention_scores, dim=-1) 
         
        # Apply attention to values 
        attended_features = torch.matmul(attention_weights, V)  # [B, N_v, D] 
         
        # Project back to visual dimension 
        output_features = self.out_proj(attended_features)  # [B, N_v, C_v] 
         
        # Reshape back to spatial format 
        output_features = output_features.permute(0, 2, 1).view(B, C_v, H_v, W_v) 
         
        # Residual connection 
        output_features = visual_features + output_features 
        output_features = self.norm(output_features) 
         
        return output_features, attention_weights 
 
Step 4:  Complete Model Architecture 
class SG_CAM_Model(nn.Module): 
    def __init__(self, num_scene_classes): 
        super().__init__() 
         
        # Dual streams 
        self.visual_extractor = VisualFeatureExtractor() 
        self.semantic_encoder = SemanticFeatureEncoder() 
         
        # Fusion module 
        self.sg_cam = SegmentationGuidedCAM( 
            visual_dim=1280, 
            semantic_dim=256, 
            hidden_dim=256 
        ) 
         
        # Classification head 
        self.classifier = nn.Sequential( 
            nn.AdaptiveAvgPool2d((1, 1)), 
            nn.Flatten(), 
            nn.Dropout(0.5), 
            nn.Linear(1280, 512), 
            nn.ReLU(), 
            nn.Linear(512, num_scene_classes) 
        ) 
         
    def forward(self, x): 
        # Extract features from both streams 
        visual_features = self.visual_extractor(x)  # [B, 1280, H/32, W/32] 
        semantic_features, seg_logits = self.semantic_encoder(x)  # [B, 256, H, W] 
         
        # Adjust semantic features resolution 
        semantic_features_resized = F.interpolate( 
            semantic_features, 
            size=visual_features.shape[-2:], 
            mode='bilinear', 
            align_corners=False 
        ) 
         
        # Apply SG-CAM fusion 
        fused_features, attention_map = self.sg_cam( 
            visual_features, 
            semantic_features_resized 
        ) 
         
        # Scene classification 
        scene_logits = self.classifier(fused_features) 
         
        return { 
            'scene_logits': scene_logits, 
            'seg_logits': seg_logits, 
            'attention_map': attention_map, 
            'fused_features': fused_features 
        } 
Step 5: Training 
def train_epoch(model, dataloader, optimizer, criterion, device): 
    model.train() 
    total_loss = 0 
      for batch_idx, (images, scene_labels, seg_masks) in enumerate(dataloader): 
        images = images.to(device) 
        scene_labels = scene_labels.to(device) 
        seg_masks = seg_masks.to(device) 
             optimizer.zero_grad() 
                # Forward pass 
        outputs = model(images) 
        scene_logits = outputs['scene_logits'] 
        seg_logits = outputs['seg_logits'] 
                # Compute losses 
        scene_loss = criterion['scene'](scene_logits, scene_labels) 
        seg_loss = criterion['seg'](seg_logits, seg_masks) 
        # Total loss (weighted combination) 
        total_batch_loss = scene_loss + 0.3 * seg_loss 
                # Backward pass 
        total_batch_loss.backward() 
        optimizer.step()   
        total_loss += total_batch_loss.item()    
return total_loss / len(dataloader) 
Step 6: Cross Dataset Evaluation 
def cross_dataset_evaluation(model, source_dataloader, target_dataloader, device): 
    """ 
    Train on source dataset (PatternNet) 
    Evaluate on target dataset (Million-AID Agricultural subset) 
    """ 
    model.eval() 
     
    source_predictions = [] 
    target_predictions = [] 
     
    with torch.no_grad(): 
        # Evaluate on source dataset 
        for images, _ in source_dataloader: 
            images = images.to(device) 
            outputs = model(images) 
            source_predictions.append(outputs['scene_logits'].cpu()) 
             
        # Evaluate on target dataset (unseen during training) 
        for images, _ in target_dataloader: 
            images = images.to(device) 
            outputs = model(images) 
            target_predictions.append(outputs['scene_logits'].cpu()) 
     
    # Calculate metrics 
    source_acc = calculate_accuracy(source_predictions) 
    target_acc = calculate_accuracy(target_predictions) 
     
    return source_acc, target_acc 
 
 
Step 7: Hyper Parameter Configuration 
 
 
config = { 
    # Training parameters 
    'batch_size': 16, 
    'learning_rate': 1e-4, 
    'epochs': 100, 
    'optimizer': 'Adam', 
    'scheduler': 'StepLR', 
     
    # Model parameters 
    'cnn_backbone': 'MobileNetV2', 
    'semantic_encoder': 'SegFormer-B2', 
    'attention_heads': 1, 
    'fusion_dim': 256, 
     
    # Data parameters 
    'image_size': 512, 
    'normalization': 'ImageNet', 
    'augmentation': ['CLAHE', 'RandomCrop', 'HorizontalFlip'], 
     
    # Loss weights 
    'scene_loss_weight': 1.0, 
    'seg_loss_weight': 0.3 
}
