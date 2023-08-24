# LIME
**The algorithm for image preprocessing in low-light conditions, named LIME, was published in the journal IEEE Transactions on Image Processing, Vol. 26, No. 2, February 2017. The title of the paper is "LIME: Low-Light Image Enhancement via Illumination Map Estimation."
# How to use?

‼️Before proceeding with the project setup, it's important to compile the OpenCV, Neon(arm only), and OpenMP libraries to ensure they are properly integrated. 
Setting up Build Directory

This repository requires a build directory to organize the compiled output. Here's how you can set it up using the terminal:

1. Open your terminal application.

2. Navigate to the root directory of this project using the `cd` command:
   ```sh
   cd path/to/your/Lime_Neon_Openmp
   mkdir build && cd build
   cmake ..
   make
   ./lime

Afterward, you will be able to observe the preprocessed images.

