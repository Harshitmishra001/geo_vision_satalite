import cv2
import numpy as np
import argparse
import json
import sys
import os
import matplotlib
# Force matplotlib to not use any Xwindows backend (Strictly Headless)
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def calculate_change(baseline_path, current_path, output_mask_path, stats_path, vis_path=None):
    """Processes two satellite images, detects terrain changes, and generates metrics/visuals."""
    
    # 1. Load the images
    img1 = cv2.imread(baseline_path)
    img2 = cv2.imread(current_path)

    if img1 is None or img2 is None:
        print(f"Error: Could not read images. Check your file paths:\n- Baseline: {baseline_path}\n- Current: {current_path}")
        sys.exit(1)

    # Ensure images are the same size for matrix operations
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    # 2. Convert to Grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 3. Compute Absolute Difference
    diff = cv2.absdiff(gray1, gray2)

    # 4. Apply Otsu's Thresholding to isolate significant changes
    _, thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # 5. Morphological Operations (Noise reduction)
    kernel = np.ones((5,5), np.uint8)
    clean_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel)

    # 6. Calculate Metrics
    total_pixels = clean_mask.shape[0] * clean_mask.shape[1]
    changed_pixels = cv2.countNonZero(clean_mask)
    percent_change = (changed_pixels / total_pixels) * 100

    # 7. Save Primary Outputs
    cv2.imwrite(output_mask_path, clean_mask)
    
    report = {
        "baseline_image": os.path.basename(baseline_path),
        "current_image": os.path.basename(current_path),
        "total_pixels_analyzed": total_pixels,
        "changed_pixels": changed_pixels,
        "percentage_change": round(percent_change, 2),
        "status": "Processing Complete"
    }

    with open(stats_path, 'w') as f:
        json.dump(report, f, indent=4)

    print(f"[SUCCESS] Change mask saved to {output_mask_path}")
    print(f"[SUCCESS] Metrics report saved to {stats_path}")
    print(f"Detected a {round(percent_change, 2)}% change in terrain.")

    # 8. Generate Visualization if requested
    if vis_path:
        print("Generating Results Visualization...")
        # Convert BGR to RGB for accurate Matplotlib color rendering
        img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        axes[0].imshow(img1_rgb)
        axes[0].set_title('Baseline Image (Before)', fontsize=16, fontweight='bold')
        axes[0].axis('off')

        axes[1].imshow(img2_rgb)
        axes[1].set_title('Current Image (After)', fontsize=16, fontweight='bold')
        axes[1].axis('off')

        axes[2].imshow(clean_mask, cmap='gray')
        axes[2].set_title('Predicted Change Mask', fontsize=16, fontweight='bold')
        axes[2].axis('off')

        plt.tight_layout()
        plt.savefig(vis_path, dpi=300, bbox_inches='tight')
        plt.close() # Clean up memory
        print(f"[SUCCESS] Visualization saved to {vis_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Headless CLI for Satellite Image Change Detection.")
    parser.add_argument('--baseline', required=True, help="Path to the older satellite image (e.g., im1.bmp)")
    parser.add_argument('--current', required=True, help="Path to the newer satellite image (e.g., im2.bmp)")
    parser.add_argument('--output', required=True, help="Path to save the generated change mask (e.g., mask.png)")
    parser.add_argument('--stats', required=True, help="Path to save the JSON metrics report (e.g., report.json)")
    parser.add_argument('--vis', required=False, help="Optional: Path to save the 3-panel visualization (e.g., results_visualization.png)")
    
    args = parser.parse_args()
    
    calculate_change(args.baseline, args.current, args.output, args.stats, args.vis)