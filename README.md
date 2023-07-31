# Development-of-terrain-mapping-using-image-processing-of-footage-captured-by-the-drone

The hexa-copter along with an attached camera is used to collect the necessary information (flight controller footage). Our aim is to obtain a stitched image from the collected footage. The main moto is to snip the similar parts of considered two images and get a single image. The semi-autonomous hexa-copter is used to capture footage and to perform a terrain analysis on the footage. Mapping the terrain over the flight path of the drone can be useful in a wide range of applications. A video footage can be converted into a panoramic image to obtain a clear view of a terrain essential for an accurate analysis. The applications include conducting of surveys for pavement of roads which would make it easy and fast than a physical survey. The main control of the hexa-copter is 'PIXHAWK PX4' controller. It contains sensors that determines the vehicle state along with an inbuilt gyroscope, accelerometer, magnetometer and a barometer. It contains its own software for configuration called 'MISSION PLANNER'. The 'MISSION PLANNER' is an open source software used to configure the rover with the GPS module and the telemetry module.
'SIFT' algorithm is applied for stitching of video footage to form a panoramic image. Taylor series and H-matrix is used for greater accuracy of images and to locate their edges appropriately. The local image gradients are measured at the selected scale in the region around each key point. Key points between two images are matched by identifying their nearest neighbour.


Working Algorithm (SIFT):

“SIFT” algorithm is applied to stitch the video footage and form a panoramic image.The algorithm identifies the localised area of interest(key points) and extracts the local invariant descriptors.The algorithm uses the difference of Gaussians which is an approximation of Laplacian of Gaussian (LoG).
Once the distance of Gaussian (DoG) is found, images are searched for local extrema over scale and space.
Taylor series and Hessian matrix is used for acute accuracy of the location of extrema and to eliminate the edges.
The local image gradients are measured at the selected scale in the region around each key point. Key points between two images are matched by identifying their nearest neighbour.
