# GeoVision CLI: Remote Sensing Terrain Change Detector 

## Overview
GeoVision is a headless Computer Vision command-line utility designed for automated remote sensing-based satellite image analysis. It compares two satellite images of the same geographical area taken at different times and generates a binary change mask highlighting structural terrain modifications. It also exports a JSON statistical report of the affected area.

This foundational classical CV pipeline serves as a robust baseline for terrain analysis, establishing a deterministic benchmark before exploring more complex deep learning architectures with spatial attention mechanisms.

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
![GeoVision Results Visualization](results\results_visualization.png)
---
## Project Report Blueprint

### Problem
Manual analysis of satellite imagery is too slow.

### Approach
A headless CLI tool was built using absolute image differencing, Otsu's Thresholding to binarize the image, and morphological opening/closing to filter out sensor noise.

### Future Scope
While this classical CV pipeline is highly efficient, future iterations could integrate deep learning models—perhaps proposing an architecture like a Nowshin net utilizing a Residual Attention module to better capture intricate spatial dependencies and contextual features in remote sensing data.

### Conclusion
The tool successfully runs in terminal environments and outputs measurable JSON metrics.
