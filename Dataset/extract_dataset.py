import tensorflow as tf
import cv2
import numpy as np
import os
import sys

def parse_raw_bytes(byte_data, is_mask=False):
    """Converts raw byte data back into a 2D/3D image matrix."""
    # First, try reading as standard 8-bit pixels
    raw_array = np.frombuffer(byte_data, dtype=np.uint8)
    total_bytes = len(raw_array)
    
    # Mathematical deduction of the image shape (assuming square satellite images)
    # Check for 3-channel RGB image
    side_length_rgb = int(np.sqrt(total_bytes / 3))
    if side_length_rgb * side_length_rgb * 3 == total_bytes:
        return raw_array.reshape((side_length_rgb, side_length_rgb, 3))
        
    # Check for 1-channel Grayscale image (common for masks)
    side_length_gray = int(np.sqrt(total_bytes))
    if side_length_gray * side_length_gray == total_bytes:
        return raw_array.reshape((side_length_gray, side_length_gray, 1))
        
    # If it's a float32 array (sometimes used in remote sensing)
    raw_float = np.frombuffer(byte_data, dtype=np.float32)
    total_floats = len(raw_float)
    side_float = int(np.sqrt(total_floats / 3))
    if side_float * side_float * 3 == total_floats:
        img = raw_float.reshape((side_float, side_float, 3))
        # Convert float (0.0 - 1.0) back to standard image (0 - 255)
        return (img * 255).astype(np.uint8)

    raise ValueError(f"Could not mathematically deduce image shape for byte length: {total_bytes}")

def extract_tfrecord(tfrec_path, output_dir="dataset", num_samples=10):
    print(f"Creating directories in {output_dir}...")
    os.makedirs(f"{output_dir}/baseline", exist_ok=True)
    os.makedirs(f"{output_dir}/current", exist_ok=True)
    os.makedirs(f"{output_dir}/ground_truth", exist_ok=True)

    feature_description = {
        'img1': tf.io.FixedLenFeature([], tf.string),
        'img2': tf.io.FixedLenFeature([], tf.string),
        'ref':  tf.io.FixedLenFeature([], tf.string),
    }

    def _parse_function(example_proto):
        return tf.io.parse_single_example(example_proto, feature_description)

    print(f"Extracting raw byte arrays from {tfrec_path}...")
    raw_dataset = tf.data.TFRecordDataset(tfrec_path)
    parsed_dataset = raw_dataset.map(_parse_function)

    extracted_count = 0
    for i, parsed_record in enumerate(parsed_dataset):
        if extracted_count >= num_samples:
            break
            
        try:
            # Extract raw bytes directly to numpy arrays using our new parser
            img1 = parse_raw_bytes(parsed_record['img1'].numpy())
            img2 = parse_raw_bytes(parsed_record['img2'].numpy())
            ref = parse_raw_bytes(parsed_record['ref'].numpy(), is_mask=True)

            # Convert RGB to BGR for OpenCV
            if img1 is not None and len(img1.shape) == 3 and img1.shape[2] == 3:
                img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
                img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)

            # Save the images
            cv2.imwrite(f"{output_dir}/baseline/pair_{i}.png", img1)
            cv2.imwrite(f"{output_dir}/current/pair_{i}.png", img2)
            cv2.imwrite(f"{output_dir}/ground_truth/pair_{i}.png", ref)
            
            extracted_count += 1
            print(f"Extracted pair {i} successfully.")
            
        except Exception as e:
            print(f"Skipping pair {i} due to decoding error: {e}")

    print(f"\n[SUCCESS] Extracted {extracted_count} image pairs to the '{output_dir}' folder!")

if __name__ == "__main__":
    extract_tfrecord("train.tfrec")