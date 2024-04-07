import argparse
import numpy as np
from PIL import Image
import tensorflow as tf  # TensorFlow Lite is now used directly

def create_pascal_label_colormap():
  """Same function as before for creating the PASCAL colormap."""
  colormap = np.zeros((256, 3), dtype=int)
  indices = np.arange(256, dtype=int)

  for shift in reversed(range(8)):
    for channel in range(3):
      colormap[:, channel] |= ((indices >> channel) & 1) << shift
    indices >>= 3

  return colormap

def label_to_color_image(label):
  """Same function as before for converting label to color image."""
  if label.ndim != 2:
    raise ValueError('Expect 2-D input label')
  colormap = create_pascal_label_colormap()
  if np.max(label) >= len(colormap):
    raise ValueError('label value too large.')
  return colormap[label]

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--model', required=True, help='Path of the segmentation model.')
  parser.add_argument('--input', required=True, help='File path of the input image.')
  parser.add_argument('--output', default='semantic_segmentation_result.jpg', help='File path of the output image.')
  parser.add_argument('--keep_aspect_ratio', action='store_true', default=False, help='Keep the image aspect ratio.')
  args = parser.parse_args()

  # Load TFLite model and allocate tensors
  interpreter = tf.lite.Interpreter(model_path=args.model)
  interpreter.allocate_tensors()

  # Get input and output details
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()
  width, height = input_details[0]['shape'][2], input_details[0]['shape'][1]

  img = Image.open(args.input)
  if args.keep_aspect_ratio:
    # Calculate aspect ratio and resize accordingly
    img_ratio = img.width / img.height
    model_ratio = width / height
    if img_ratio >= model_ratio:
      new_height = int(img.width / model_ratio)
      resized_img = img.resize((img.width, new_height), Image.LANCZOS)
    else:
      new_width = int(img.height * model_ratio)
      resized_img = img.resize((new_width, img.height), Image.LANCZOS)
  else:
    resized_img = img.resize((width, height), Image.LANCZOS)

  input_data = np.expand_dims(resized_img, axis=0)
  input_data = np.array(input_data, dtype=np.float32)

  # Set the tensor to point to the input data to be inferred
  interpreter.set_tensor(input_details[0]['index'], input_data)
  interpreter.invoke()

  # Get the model prediction
  result = interpreter.get_tensor(output_details[0]['index'])[0]
  if len(result.shape) == 3:
    result = np.argmax(result, axis=-1)

  # Convert the output to an image
  mask_img = Image.fromarray(label_to_color_image(result).astype(np.uint8))

  # Save the output image
  mask_img.save(args.output)
  print('Done. Results saved at', args.output)

if __name__ == '__main__':
  main()
