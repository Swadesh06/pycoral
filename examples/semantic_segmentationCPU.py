import argparse
import numpy as np
from PIL import Image
import tensorflow as tf  # Ensure TensorFlow is installed
import time 

def create_pascal_label_colormap():
    """Creates a label colormap used in PASCAL VOC segmentation benchmark."""
    colormap = np.zeros((256, 3), dtype=int)
    indices = np.arange(256, dtype=int)

    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((indices >> channel) & 1) << shift
        indices >>= 3

    return colormap

def label_to_color_image(label):
    """Adds color defined by the dataset colormap to the label."""
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
    parser.add_argument('--keep_aspect_ratio', action='store_true', default=False, help='Keep the image aspect ratio when down-sampling the image by adding black pixel padding on bottom or right.')
    args = parser.parse_args()

    interpreter = tf.lite.Interpreter(model_path=args.model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Assume the model expects input images in the form [1, height, width, 3]
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    img = Image.open(args.input).convert('RGB')  # Convert to RGB
    original_size = img.size
    resample_method = Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.ANTIALIAS

    if args.keep_aspect_ratio:
        # Calculate new size preserving aspect ratio
        aspect_ratio = img.width / img.height
        if width / height > aspect_ratio:
            new_height = height
            new_width = int(new_height * aspect_ratio)
        else:
            new_width = width
            new_height = int(new_width / aspect_ratio)
        img = img.resize((new_width, new_height), resample_method)
        new_img = Image.new("RGB", (width, height))
        new_img.paste(img, ((width - new_width) // 2, (height - new_height) // 2))
        img = new_img
    else:
        img = img.resize((width, height), resample_method)
    
    input_data = np.expand_dims(img, axis=0)
    if input_details[0]['dtype'] == np.float32:
        input_data = input_data.astype(np.float32) / 255.0  # Normalize if needed
        
    start_time = time.time()
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    end_time = time.time()

    inference_time = (end_time - start_time) * 1000
    print(f"Inference Time: {inference_time:.2f} ms")


    result = interpreter.get_tensor(output_details[0]['index'])[0]
    if len(result.shape) == 3:  # For models outputting a 3D array
        result = np.argmax(result, axis=-1)

    # Resize mask to match original image size
    mask_img = Image.fromarray(label_to_color_image(result).astype(np.uint8))
    mask_img = mask_img.resize(original_size, Image.NEAREST)  # Resize to original size

    # Instead of concatenating resized input (which might be padded or cropped), use the original image
    output_img = Image.new('RGB', (original_size[0] * 2, original_size[1]))
    output_img.paste(img, (0, 0))  # Paste the original image, not the resized one
    output_img.paste(mask_img, (original_size[0], 0))  # Paste next to the original image
    output_img.save(args.output)
    print('Done. Results saved at', args.output)

if __name__ == '__main__':
    main()
