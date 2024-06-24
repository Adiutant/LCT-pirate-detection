import torch.nn.functional as F
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import ViTModel, ViTFeatureExtractor, TrainingArguments, Trainer
from datasets import load_metric
import numpy as np
import cv2


# class CustomImageDataset(Dataset):
#     def __init__(self, images, labels, feature_extractor):
#         self.images = images
#         self.labels = labels
#         self.feature_extractor = feature_extractor
#
#     def __len__(self):
#         return len(self.images)
#
#     def __getitem__(self, idx):
#         image_1 = self.images[idx][0]
#         image_2 = self.images[idx][1]
#         label = self.labels[idx]
#         inputs_1 = self.feature_extractor(images=image_1, return_tensors="pt")
#         inputs_2 = self.feature_extractor(images=image_2, return_tensors="pt")
#         return {
#             "pixel_values_1": inputs_1["pixel_values"].squeeze(),
#             "pixel_values_2": inputs_2["pixel_values"].squeeze(),# Remove batch dimension
#             "labels": torch.tensor(label, dtype=torch.long)
#         }

class ViTForImageClassification(torch.nn.Module):
    def __init__(self, num_classes=2):
        super(ViTForImageClassification, self).__init__()
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        self.classifier = torch.nn.Linear(self.vit.config.hidden_size * 2, num_classes)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, pixel_values):
        pixel_values = pixel_values.to(self.device)
        outputs = self.vit(pixel_values=pixel_values)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return cls_embedding.detach().cpu().numpy()

    def classify(self, embeddings_1, embeddings_2, labels=None):
        # pixel_values_1 = pixel_values_1.to(self.device)
        # pixel_values_2 = pixel_values_2.to(self.device)
        # if labels is not None:
        #     labels = labels.to(self.device)
        # outputs_1 = self.vit(pixel_values=pixel_values_1)
        # outputs_2 = self.vit(pixel_values=pixel_values_2)

        # # Extract the embeddings of the CLS token from both outputs
        # cls_embedding_1 = outputs_1.last_hidden_state[:, 0, :]  # CLS token
        # cls_embedding_2 = outputs_2.last_hidden_state[:, 0, :]  # CLS token

        embeddings_1 = torch.from_numpy(embeddings_1).to(self.device)
        embeddings_2 = torch.from_numpy(embeddings_2).to(self.device)

        # Concatenate the embeddings
        combined_embeddings = torch.cat((embeddings_1, embeddings_2), dim=1)

        # Pass the combined embeddings through the classifier
        logits = self.classifier(combined_embeddings)

        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            return loss, logits

        return logits.detach().cpu().numpy()
