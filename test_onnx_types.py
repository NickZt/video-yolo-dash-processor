import onnxruntime as ort
session = ort.InferenceSession('test_assets/groundingdino.onnx')
for i in session.get_inputs():
    print(f"Name: {i.name}, Type: {i.type}")
