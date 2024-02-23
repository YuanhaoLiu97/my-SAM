import time
import torch
import numpy as np
import cv2
from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.onnx import SamOnnxModel
import onnxruntime
from onnxruntime.quantization import QuantType
from onnxruntime.quantization.quantize import quantize_dynamic
import warnings




checkpoint = "../model/sam_vit_l_0b3195.pth"
model_type = "vit_l"
sam = sam_model_registry[model_type](checkpoint=checkpoint)
export_onnx = True
if export_onnx:
    onnx_model_path = "../model/sam_onnx_vit_l_0b3195.onnx"

    onnx_model = SamOnnxModel(sam, return_single_mask=True)

    dynamic_axes = {
        "point_coords": {1: "num_points"},
        "point_labels": {1: "num_points"},
    }

    embed_dim = sam.prompt_encoder.embed_dim
    embed_size = sam.prompt_encoder.image_embedding_size
    mask_input_size = [4 * x for x in embed_size]
    dummy_inputs = {
        "image_embeddings": torch.randn(1, embed_dim, *embed_size, dtype=torch.float),
        "point_coords": torch.randint(low=0, high=1024, size=(1, 5, 2), dtype=torch.float),
        "point_labels": torch.randint(low=0, high=4, size=(1, 5), dtype=torch.float),
        "mask_input": torch.randn(1, 1, *mask_input_size, dtype=torch.float),
        "has_mask_input": torch.tensor([1], dtype=torch.float),
        "orig_im_size": torch.tensor([1500, 2250], dtype=torch.float),
    }
    output_names = ["masks", "iou_predictions", "low_res_masks"]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        with open(onnx_model_path, "wb") as f:
            torch.onnx.export(
                onnx_model,
                tuple(dummy_inputs.values()),
                f,
                export_params=True,
                verbose=False,
                opset_version=17,
                do_constant_folding=True,
                input_names=list(dummy_inputs.keys()),
                output_names=output_names,
                dynamic_axes=dynamic_axes,
            )    

export_emmbeding = False
if export_emmbeding:
    image = cv2.imread('../src/assets/data/0006.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    sam.to(device='cpu')
    predictor = SamPredictor(sam)

    # 计算 predict 的时间
    start_predict_time = time.time()
    predictor.set_image(image)
    end_predict_time = time.time()
    predict_time = end_predict_time - start_predict_time
    print("Predict 时间（秒）:", predict_time)

    # 计算 embedding 的时间
    start_embedding_time = time.time()
    image_embedding = predictor.get_image_embedding().cpu().numpy()
    end_embedding_time = time.time()
    embedding_time = end_embedding_time - start_embedding_time
    print("Embedding 时间（秒）:", embedding_time)

    print(image_embedding.shape)
    with open('../src/assets/data/embedding.npy', 'wb') as f:
        np.save(f, image_embedding)