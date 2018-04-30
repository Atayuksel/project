# project
The development is done in Windows 8.1 OS.
The developement environment is Python 3.6.1, CUDA version: 9.1, Cuda Compute Capability Major/Minor v: 3.5, MKL 2018.2.185. I used 'cmder' as a command prompt in Windows to ease development process. I used Nvidia Cuda Compiler (NVCC) for compiling and linking of c files.
Library Paths:
MKL: C:\Program Files (x86)\IntelSWTools\compilers_and_libraries_2018.2.185
CUDA: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.1
Hardware:
Processor: Intel Core i7-4702MQ
GPU: Nvidia Geforce 740M

Batch File:
1- I build compiled files in /build folder thus in order to accelerate testing process I delete /build folder in every build. Then I create a new /build folder and create compiled files there. 

2- I compile the cuda files with NVCC. The -x cu parameter defines the language of the file that will be compiled. The parameters arch and code defines the virtual and real architecture of the GPU device. In my case, these options are sm_35 and compute_35 because of my own GPU device. In order to get compute capacity of the device, devicequery.exe can be used which is installed with CUDA toolkit automatically. In order to compile .cu file to relocatable device code, I used -dc option. We need to use this parameter. Because, I used cuBlas library which is pre-compile cuda library and in order to link with cublas without any error we need to compile our own cuda file as relocatable device code. 

3- In linking process, we want to create a shared library to be used in Python as a module thus we add shared options. We also add arch and code options too as we did in compile step. Then we add required libraries and specifiy output file as .pyd file to be used in Pyhton.

4- Finally, we copy our test python script to /build folder and run the script in /build.

In order to run the cuda files correctly, libraries should be installed correctly. cuda_exp.cu, test.py and make.bat file should be located in same directory. User should have admin privileges to delete and create folders. 
Some changes should be done in .bat file;
- need to change paths based on the correct locations in your own computer
- need to change -arch and -code options based on your own GPU device and in order to do that devicequery.exe can be used.
(In my environment location of devicequery is "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.1\extras\demo_suite")

Then compile and link process should be run flawless.
