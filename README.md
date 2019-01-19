# Project

### Environment
- Operating System: Windows 8.1
- Developement environment is Python 3.6.1, CUDA version: 9.1, Cuda Compute Capability Major/Minor v: 3.5, MKL 2018.2.185. I used 'cmder' as a command prompt in Windows to ease development process. I used Nvidia Cuda Compiler (NVCC) for compiling and linking of c files.
- Library Paths:
  - MKL: C:\Program Files (x86)\IntelSWTools\compilers_and_libraries_2018.2.185
  - CUDA: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.1
- Hardware:
  - Processor: Intel Core i7-4702MQ
  - GPU: Nvidia Geforce 740M

### Batch File
1. I build compiled files in /build folder thus in order to accelerate testing process I delete /build folder in every build. Then I create a new /build folder and create compiled files there. 
2. I compile the cuda files with NVCC. The -x cu parameter defines the language of the file that will be compiled. The parameters arch and code defines the virtual and real architecture of the GPU device. In my case, these options are sm_35 and compute_35 because of my own GPU device. In order to get compute capacity of the device, devicequery.exe can be used which is installed with CUDA toolkit automatically. In order to compile .cu file to relocatable device code, I used -dc option. We need to use this parameter. Because, I used cuBlas library which is pre-compile cuda library and in order to link with cublas without any error we need to compile our own cuda file as relocatable device code. 
3. In linking process, we want to create a shared library to be used in Python as a module thus we add shared options. We also add arch and code options too as we did in compile step. Then we add required libraries and specifiy output file as .pyd file to be used in Pyhton.
4. Finally, we copy our test python script to /build folder and run the script in /build.

### Changes to compile, link and test
In order to run the cuda files correctly, libraries should be installed correctly. cuda_exp.cu, test.py and make.bat file should be located in same directory. User should have admin privileges to delete and create folders. 
Some changes should be done in .bat file;
- need to change paths based on the correct locations in your own computer
```
nvcc -x cu -arch=your-own-cc -code=your-own-cc -dc cuda_exp.cu -I"Python include path" -I"MKL include path" -I"CUDA includepath" -I"numpy include path" -o build\cuda_exp.obj
```
```
nvcc -shared -arch=your-own-cc -code=your-own-cc build\cuda_exp.obj -L"Python libs path" -L"MKL libs path" -L"CUDA libs path" -lcublas -lcublas_device -lcudadevrt -lcudart_static -lmkl_intel_lp64_dll -lmkl_sequential_dll -lmkl_core_dll -o build\wordtovec.pyd
```
- In order to get your-own-cc, you could run devicequery.exe, it is installed with CUDA toolkit automatically. <br />
In my case, it is located at "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.1\extras\demo_suite" <br />

Then compile and link process should be run flawless.

### driver.py
The python script is used to parse the dataset and create more structured data. Currently, it is only used to parse the data and create input list to wordtovec algorithm. There are some missing parts in script file. I aimed to used numpy arrays in the beginning, when I started to implement wordtovec algorithm in c in order to speed up the process. Currently, the script outputs native python arrays. I use numpy arrays as I planned in test.py file. This will not be very time consuming task to do.

### Problems
I think, I'm facing with a race condition in the wordtovec implementation. Currently, I'm trying to understand and solve the problem. In small sets like test.py, there are not any problems. In real dataset, I faced with a problem. <br />
I will also try to implement basic LSTM and CNN architectures in CUDA and compare the results. 

### Sources
- https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/
- https://docs.nvidia.com/cuda/cublas/
- https://devblogs.nvidia.com/even-easier-introduction-cuda/
- https://docs.python.org/3/extending/extending.html
- http://sep.stanford.edu/sep/claudio/Research/Prst_ExpRefl/ShtPSPI/intel/mkl/10.0.3.020/examples/
- https://software.intel.com/en-us/mkl-developer-reference-c
