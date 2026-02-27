import os
from onnxruntime.quantization import quantize_dynamic, QuantType

def quantize_existing_model():
    onnx_path = "test_assets/groundingdino.onnx"
    quantized_path = "test_assets/groundingdino_int8.onnx"
    
    if not os.path.exists(onnx_path):
        print(f"Error: Could not find baseline model at {onnx_path}")
        return

    print(f"Applying ONNX Dynamic INT8 Quantization to {onnx_path}...")
    
    # Apply Dynamic Quantization natively using ONNX Runtime
    quantize_dynamic(
        model_input=onnx_path,
        model_output=quantized_path,
        weight_type=QuantType.QUInt8,
        # Only quantize dense matrix ops where INT8 yields maximum CPU throughput
        op_types_to_quantize=['MatMul', 'Add'] 
    )
    print(f"Quantized INT8 model successfully saved to {quantized_path}")

if __name__ == "__main__":
    quantize_existing_model()
