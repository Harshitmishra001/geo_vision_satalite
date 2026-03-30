# GeoVision CLI: Remote Sensing Terrain Change Detector 

## Overview
GeoVision is a command-line computer vision tool built for automated analysis of satellite imagery. It takes two images of the same geographic region captured at different points in time and compares them to identify changes. The output includes a binary change mask that highlights structural modifications in the terrain, along with a JSON report summarizing the extent of the affected area.

At its core, the system relies on a classical computer vision pipeline, providing a stable and interpretable baseline for terrain analysis. This deterministic approach is useful for establishing a clear benchmark, especially before moving toward more advanced deep learning models that incorporate spatial attention and other complex mechanisms.

## Prerequisites
- Python 3.8+
- No GUI dependencies required (fully headless execution).

## Installation
1. Clone the repository:
   git clone https://github.com/Harshitmishra001/geo_vision_satalite
   cd geo-vision

2. Install the required dependencies:
   pip install opencv-python numpy

## Usage
The application is executed entirely via the command line.

### Example Execution (Using included SZTAKI Benchmark Data):
python analyze_terrain.py --baseline SZTAKI_AirChange_Benchmark/Szada/1/im1.bmp --current SZTAKI_AirChange_Benchmark/Szada/1/im2.bmp --output predicted_change_mask.png --stats report.json --vis results_visualization.png

## Output Artifacts
- Change Mask (.png): A binary image where white pixels represent isolated changes.
- Metrics Report (.json): A data file detailing the exact pixel count and percentage of terrain alteration.

---

## GitHub Iterative Commits
The repository should reflect an actual development process. Use the following commands:

git init
git add README.md
git commit -m "docs: Initialize project scope and headless CLI instructions"

git add analyze_terrain.py
git commit -m "feat: Implement OpenCV image differencing and Otsu thresholding"

git add SZTAKI_AirChange_Benchmark/Szada/1/
git commit -m "test: Add SZTAKI benchmark sample pair for automated evaluation"

git branch -M main
git remote add origin https://github.com/Harshitmishra001/geo_vision_satalite
git push -u origin main

---
### Results
![GeoVision Results Visualization](results/results_visualization.png)
---
## Project Report Blueprint

### Problem
Manual analysis of satellite imagery is too slow.

### Approach
A headless command line tool was built around a simple classical pipeline. It starts by comparing two images using absolute differencing to highlight pixel level changes. The result is then converted into a binary image through Otsu’s thresholding. After that, morphological opening and closing are applied to reduce sensor noise and clean up the final output.

### Future Scope
While this classical CV pipeline is highly efficient, future iterations could integrate deep learning models—perhaps proposing an architecture like a Nowshin net utilizing a Residual Attention module to better capture intricate spatial dependencies and contextual features in remote sensing data.

### Conclusion
The tool successfully runs in terminal environments and outputs measurable JSON metrics.
