# Automated Vehicle Damage Detection (Multimodal Learning)
This repository contains the code, a small sample of cropped ROI images, and metadata for an automated vehicle damage detection pipeline.  
The project combines **computer vision** and **text metadata** to detect and classify vehicle damage, analogous to autonomous vehicle (AV) perception evaluation.

The system is designed to:
- Detect red bounding boxes around damaged regions in vehicle images.
- Crop the region of interest (ROI) for targeted analysis.
- Classify damage type and location using **deep learning**.
- Integrate both **image features** and **text metadata** for improved accuracy.

Two main model variants are implemented:
1. **Vision-only** (VGG16 fine-tuning)
2. **Multimodal** (CLIP text–image embeddings), which **outperformed** the vision-only baseline.


