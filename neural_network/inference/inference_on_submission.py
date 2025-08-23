from neural_network.configuration import systemConfig, inferenceConfig
from neural_network.data.data_loader import get_indexes_to_class_mapping
from neural_network.data.augmentation import get_resize_pipeline

import pandas as pd
from PIL import Image
import numpy as np
from tqdm import tqdm
import torch

from torchvision import transforms
import os


def score_submission(model):
    if not inferenceConfig.score_submission:
        return

    image_ids = _get_image_indexes()

    predictions = []

    model.eval()
    with torch.no_grad():
        for img_id in tqdm(image_ids, desc="Predicting test set"):
            predicted_label = _run_inference_on_image(img_id, model)
            predictions.append(predicted_label)

    _save_predictions_to_csv(image_ids, predictions)



def _get_image_indexes():
    submission_empty_dataset = pd.read_csv(inferenceConfig.submission_label_file)
    image_ids = submission_empty_dataset.iloc[:, 0].tolist()
    return image_ids

def _run_inference_on_image(img_name, model):
    indexes_to_class = get_indexes_to_class_mapping()

    img_path = _get_image_path(img_name)
    image = _prepare_image_for_inference(img_path)
    output = model(image)
    predicted_class_index = output.argmax(dim=1).item()
    predicted_class = indexes_to_class[predicted_class_index]
    return predicted_class


def _get_image_path(img_name):
    submission_image_directory = inferenceConfig.submission_image_directory

    img_name = str(img_name)
    if not img_name.lower().endswith('.jpg'):
        img_name += '.jpg'  
    return os.path.join(submission_image_directory, img_name)

def _prepare_image_for_inference(img_path):
    resize_augmentation_pipeline = get_resize_pipeline()

    image = Image.open(img_path).convert('RGB')
    image = np.array(image)
    image = resize_augmentation_pipeline(image=image)['image']
    image = transforms.ToTensor()(image)
    image = image.unsqueeze(0).to(systemConfig.device)
    return image

def _save_predictions_to_csv(image_ids, predictions):
    path = inferenceConfig.inference_on_submission_output_path
    os.makedirs(path, exist_ok=True)
    file = inferenceConfig.inference_on_submission_output_file
    filepath = os.path.join(path, file)

    submission_df = pd.DataFrame({
        'id': image_ids,
        'label': predictions
    })

    submission_df.to_csv(filepath, index=False)
    print(f'Saved predictions on submission to {filepath}.')