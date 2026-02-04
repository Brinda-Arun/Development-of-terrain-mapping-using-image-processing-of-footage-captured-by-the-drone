# Terrain Mapping Using Drone Footage and Image Stitching

This project focuses on generating a panoramic terrain map by stitching aerial video footage captured using a semi-autonomous hexa-copter equipped with a camera. The stitched panoramic image provides a clear and continuous view of the terrain, enabling accurate analysis and mapping.

## Project Overview
A hexa-copter integrated with a flight controller and camera is used to collect aerial footage along a predefined flight path. The captured video is processed to extract overlapping frames, which are then stitched together to form a single panoramic image of the terrain.

The primary goal is to identify and merge similar regions from consecutive frames to create a seamless terrain map.

## Hardware and Software Setup
- **Drone:** Semi-autonomous hexa-copter  
- **Flight Controller:** PIXHAWK PX4  
- **Sensors:** Gyroscope, accelerometer, magnetometer, barometer  
- **Configuration Software:** Mission Planner (open-source)  
- **Input Data:** Aerial video footage captured during flight  

## Methodology
- Convert aerial video footage into image frames
- Identify overlapping regions between consecutive frames
- Stitch images to generate a panoramic terrain view
- Perform terrain analysis based on the stitched output

## Image Stitching Technique
The project uses the **Scale-Invariant Feature Transform (SIFT)** algorithm to identify and match key points between overlapping images.

### Working Algorithm (SIFT)
- Detect scale-invariant key points using Difference of Gaussians (DoG), an approximation of Laplacian of Gaussian (LoG)
- Locate extrema across scale and spatial dimensions
- Refine key point localization using Taylor series expansion and Hessian matrix
- Eliminate edge responses to improve accuracy
- Compute local image gradients around each key point
- Match key points between images using nearest-neighbor matching
- Align and stitch images using homography (H-matrix)

## Applications
- Terrain mapping and analysis
- Road and pavement survey planning
- Aerial surveying for infrastructure development
- Faster and more efficient alternative to manual field surveys

## Technologies Used
- Python
- OpenCV
- SIFT Algorithm
- Image Proc

