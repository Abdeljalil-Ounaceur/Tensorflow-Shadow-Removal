import torch
import torch.onnx
from model import StyleGAN, BiSeNet 

stylegan_model = torch.load('checkpoint/550000.pt', map_location=torch.device('cpu'))
face_seg_model = torch.load('checkpoint/face-seg-BiSeNet-79999_iter.pth', map_location=torch.device('cpu'))

stylegan_model.eval()
face_seg_model.eval()

dummy_input = torch.randn(1, 3, 256, 256)

try:
    torch.onnx.export(
        stylegan_model,           # model being run
        dummy_input,              # model input (or a tuple for multiple inputs)
        'stylegan.onnx',          # where to save the model
        export_params=True,        # store the trained parameter weights inside the model file
        opset_version=12,          # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=['input'],     # the model's input names
        output_names=['output'],   # the model's output names
        dynamic_axes={             # optional: mark input/output axes as dynamic
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )

    torch.onnx.export(
        face_seg_model,
        dummy_input,
        'face_seg.onnx',
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )

    print("Models have been converted to ONNX format successfully.")

except Exception as e:
    print(f"Export failed: {e}")
    try:
        traced_stylegan = torch.jit.trace(stylegan_model, dummy_input)
        torch.jit.save(traced_stylegan, 'stylegan.onnx')
        
        traced_face_seg = torch.jit.trace(face_seg_model, dummy_input)
        torch.jit.save(traced_face_seg, 'face_seg.onnx')
        
        print("Models exported using torch.jit.trace")
    except Exception as trace_e:
        print(f"Tracing failed: {trace_e}")