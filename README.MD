## Introduction
This sample was created to demonstrate and measure speed of searching for max element and its index in CUDA with different algorithms comparing to C++ STL max_element function.
## Build
 * Download and install [Visual Studio 2017](https://www.visualstudio.com/products/visual-studio-community-vs)
 * Download and install CUDA
   * Download [CUDA Toolkit 10.0](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exenetwork)
   * Launch the downloaded installer package
   * Read and accept the EULA
   * Select "Next" to download and install all components
   * Once the download completes, the installation will begin automatically
   * Follow the steps of installation wizard
   * Finish installation 
 * Download and install [CMake](https://cmake.org/download/)
 * Get the sources
    * `git clone https://github.com/apriorit/cuda-reduce-max-with-index.git` into `c:\cuda-reduce-max-with-index` (destination dir can be anything, `c:\cuda-reduce-max-with-index` is used here just for the reference)
 * Generate Visual Studio project files
    * Launch CMake (cmake-gui) from the start menu
    * Set "Where is the source code" to `c:\cuda-reduce-max-with-index`
    * Set "Where to build the binaries" to `c:\cuda-reduce-max-with-index\bin`
    * Click "Generate"
    * Choose "Visual Studio 15 2017 Win64"
    * Click "Finish"
    * Click "Open Project" or launch (may not be available for CUDA projects) `c:\cuda-reduce-max-with-index\bin\cuda-reduce-max-with-index.sln`
 * Build
    * Set Solution Configurations
    * Choose from the main menu "Build->Build solution"
    * Navigate to `c:\cuda-reduce-max-with-index\bin\<configuration output folder>` and get `ReduceMaxWithIndex.exe`
 * Build with msbuild.exe
    * `msbuild.exe bin/ReduceMaxWithIndex.exe.sln /t:Build /p:Configuration=Release`