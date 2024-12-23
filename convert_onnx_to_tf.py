from onnx_tf.backend import prepare
import onnx

# Load the ONNX models
stylegan_onnx = onnx.load('stylegan.onnx')
face_seg_onnx = onnx.load('face_seg.onnx')

# Convert the ONNX models to TensorFlow
stylegan_tf = prepare(stylegan_onnx)
face_seg_tf = prepare(face_seg_onnx)

# Save the TensorFlow models
stylegan_tf.export_graph('stylegan_tf')
face_seg_tf.export_graph('face_seg_tf')

print("Models have been converted to TensorFlow format.")

