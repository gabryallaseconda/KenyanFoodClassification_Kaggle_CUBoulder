from neural_network.configuration import modelConfig, dataConfig
from datetime import datetime
import torch


def export_to_pth(model):

    filepath = _get_filepath("pth")

    torch.save(model.state_dict(), filepath)
    print(f"Model exported with format torch to {filepath}.")

def export_to_onnx(model):
    dummy_input = torch.randn(input_size=(1, 3, dataConfig.resized_image_width, dataConfig.resized_image_height), 
                              device=modelConfig.device)
    
    filepath = _get_filepath("onnx")

    torch.onnx.export(model, 
                      dummy_input, 
                      filepath, 
                      export_params=True, 
                      opset_version=11, 
                      do_constant_folding=True,
                      input_names=['input'], 
                      output_names=['output'])
    print(f"Model exported with format onnx to {filepath}.")



def _get_filepath(extension: str) -> str:
    import os

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    path = modelConfig.model_saving_path
    os.makedirs(path, exist_ok=True)

    prefix = modelConfig.model_prefix
    filepath = os.path.join(path, prefix)

    return f"{filepath}_{timestamp}.{extension}"