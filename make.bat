if exist build rmdir build /s /q
mkdir build
nvcc -x cu -arch=compute_35 -code=sm_35 -dc cuda_exp.cu -IC:\Users\Atakan\AppData\Local\Programs\Python\Python36\include -I"C:\Program Files (x86)\IntelSWTools\compilers_and_libraries_2018.2.185\windows\mkl\include" -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.1\include" -I"C:\Users\Atakan\AppData\Local\Programs\Python\Python36\lib\site-packages\numpy\core\include" -o build\cuda_exp.obj
nvcc -shared -arch=compute_35 -code=sm_35 build\cuda_exp.obj -L"C:\Users\Atakan\AppData\Local\Programs\Python\Python36\libs" -L"C:\Program Files (x86)\IntelSWTools\compilers_and_libraries_2018.2.185\windows\mkl\lib\intel64" -L"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.1\lib\x64" -lcublas -lcublas_device -lcudadevrt -lcudart_static -lmkl_intel_lp64_dll -lmkl_sequential_dll -lmkl_core_dll -o build\wordtovec.pyd
xcopy test.py build
cd build
python test.py