# Custom Object Detection with TensorFlow-2

These are the steps i took to train an Object Detection model to detect my face. I used TensorFlow 2 Object Detection API on Windows 10. I used a virtual machine on Microsoft Azure. My VM was Standard DS2 v2, with 2 virtual CPUs and 7 GB of memory.

## Softwares needed
- [Python](https://www.python.org/downloads/windows/) [enable long paths in the setup]
- [git](https://git-scm.com/downloads)
- [Anaconda](https://www.anaconda.com/products/individual) [set this as PATH in the setup]
- [Visual Studio 2019 with C++ Build Tools](https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=Community&rel=16)
- [Visual C++ 2015 build tools](https://go.microsoft.com/fwlink/?LinkId=691126)
- [LabelImg](https://tzutalin.github.io/labelImg/)
- [7Zip](https://www.7-zip.org/download.html)

## Installing TensorFlow CPU
Open the anaconda command prompt. Create a virtual environment
```
conda create -n tensorflow pip python=3.8
```
Then activate the environment with

```
conda activate tensorflow
```
**NOTE: The virtual environment has to be activated each time the anaconda terminal is closed.**
Installing TensorFlow CPU
```
pip install tensorflow
```
Sanity Check
```
python
>>> import tensorflow as tf
>>> print(tf.__version__)
>>> exit()
```
A version number should be displayed.

### Preparing our Workspace
Create a folder directly in C: and name it "TensorFlow". It can be created anywhere but the commands need to be changed accordingly.

```
cd C:\TensorFlow
```
Clone the [TensorFlow models repository](https://github.com/tensorflow/models) with

```
git clone https://github.com/tensorflow/models.git
```
This should clone all the files in a directory called models. After this stay inside C:\TensorFlow and download [this](https://github.com/Purefekt/Custom-Object-Detection-with-TensorFlow-2) repository into a .zip file. Then extract the two files, workspace and scripts, highlighted below directly in to the TensorFlow directory.
<p align="left">
  <img src="Assets/Download zip.png" = 250x250>
</p>
