#
# Notes on installing Keras & Tensorflow, with GPU support, on Ubuntu 16.04
#
# This is NOT a runnable script, since your setup may differ in important ways.
# lines prefixed with "robot" are for hetzner.com hosting; "gcloud" are for google hosting.
#

# robot: fix apt-get for ipv4.
# https://askubuntu.com/questions/620317/apt-get-update-stuck-connecting-to-security-ubuntu-com
echo 'precedence ::ffff:0:0/96 100' >> /etc/gai.conf

# robot: assign a hostname - gcloud does this automatically
sudo hostname $hostname

# robot: setup a user account - gcloud does this automatically if you provide the SSH key in the gcloud console, which I recommend
echo "setting up asah user account..."
useradd -m -s /bin/bash asah; echo -e "asah\tALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/50_asah
asahssh=/home/asah/.ssh
mkdir $asahssh; chown asah $asahssh; chgrp asah $asahssh; chmod 700 $asahssh
echo <...contents of id_rsa.pub...>   > $asahssh/authorized_keys; chown asah $asahssh/authorized_keys; chgrp asah $asahssh/authorized_keys; chmod 600 $asahssh/authorized_keys

# install NVIDIA CUDA drivers
sudo ./cuda_8.0.61_375.26_linux-run
# hand-copy the cudnn drivers from NVIDIA - https://developer.nvidia.com/cudnn
# IMPORTANT: since /usr/local/cuda is a symlink, you don't want to copy 
sudo cp -r cuda/include cuda/lib64 /usr/local/cuda

# before anything else, always do this
apt-get -y update; apt-get -y upgrade

# adam's personal preferences
sudo apt-get -y install emacs aptitude

# for compiling tensorflow from sources
sudo apt-get -y install make gcc openjdk-8-jdk git python-dev python3-dev python-numpy python3-numpy python-six python3-six build-essential python-pip python3-pip python-virtualenv swig python-wheel python3-wheel libcurl3-dev libcupti-dev zip

# bazel: see https://bazel.build/
echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -
sudo apt-get -y update && sudo apt-get -y install bazel && sudo apt-get -y upgrade bazel

git clone https://github.com/tensorflow/tensorflow
cd tensorflow
echo "***"
echo "*** see instructions for recommended answers to tensorflow config questions"
echo "*** for NVIDIA version question, see https://developer.nvidia.com/cuda-gpus - K80 is 3.7, GTX 1080 is 6.1"
echo "*** for CUDNN version question, provide the complete version e.g. 5.1.10
echo "***"
./configure
bazel build --local_resources 50000,8.0,10.0 --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package; bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
sudo pip install /tmp/tensorflow_pkg/tensorflow-*.whl
sudo pip install h5py keras  # I've heard reports that keras likes to be installed after tensorflow...

echo "testing tensorflow... should see hello world"
# note: errors if run from tensorflow source directory, due to python imports
cd /tmp; python -c 'import tensorflow as tf;hello = tf.constant("Hello world, from TensorFlow.");sess = tf.Session();print(sess.run(hello))'


---

### NVIDIA CUDA installation - note that errors may be ok !!!

Do you accept the previously read EULA?
accept/decline/quit: accept

Install NVIDIA Accelerated Graphics Driver for Linux-x86_64 375.26?
(y)es/(n)o/(q)uit: y

Do you want to install the OpenGL libraries?
(y)es/(n)o/(q)uit [ default is yes ]:

Do you want to run nvidia-xconfig?
This will update the system X configuration file so that the NVIDIA X driver
is used. The pre-existing X configuration file will be backed up.
This option should not be used on systems that require a custom
X configuration, such as systems with multiple GPU vendors.
(y)es/(n)o/(q)uit [ default is no ]:

Install the CUDA 8.0 Toolkit?
(y)es/(n)o/(q)uit: y

Enter Toolkit Location
 [ default is /usr/local/cuda-8.0 ]:

Do you want to install a symbolic link at /usr/local/cuda?
(y)es/(n)o/(q)uit: y

Install the CUDA 8.0 Samples?
(y)es/(n)o/(q)uit: n


Installing the NVIDIA display driver...
The driver installation has failed due to an unknown error. Please consult the driver installation log located at /var/log/nvidia-installer.log.

===========
= Summary =
===========

Driver:   Installation Failed
Toolkit:  Installation skipped
Samples:  Not Selected



#---------------------------------------------------------------------------
### Tensorflow ./configure answers

.........
You have bazel 0.5.1 installed.
Please specify the location of python. [Default is /usr/bin/python]:
Found possible Python library paths:
  /usr/local/lib/python2.7/dist-packages
  /usr/lib/python2.7/dist-packages
Please input the desired Python library path to use.  Default is [/usr/local/lib/python2.7/dist-packages]

Using python library path: /usr/local/lib/python2.7/dist-packages
Do you wish to build TensorFlow with MKL support? [y/N] y
MKL support will be enabled for TensorFlow
Do you wish to download MKL LIB from the web? [Y/n] y
Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified [Default is -march=native]:
Do you wish to use jemalloc as the malloc implementation? [Y/n]
jemalloc enabled
Do you wish to build TensorFlow with Google Cloud Platform support? [y/N]
No Google Cloud Platform support will be enabled for TensorFlow
Do you wish to build TensorFlow with Hadoop File System support? [y/N]
No Hadoop File System support will be enabled for TensorFlow
Do you wish to build TensorFlow with the XLA just-in-time compiler (experimental)? [y/N]
No XLA support will be enabled for TensorFlow
Do you wish to build TensorFlow with VERBS support? [y/N]
No VERBS support will be enabled for TensorFlow
Do you wish to build TensorFlow with OpenCL support? [y/N]
No OpenCL support will be enabled for TensorFlow
Do you wish to build TensorFlow with CUDA support? [y/N] y
CUDA support will be enabled for TensorFlow
Do you want to use clang as CUDA compiler? [y/N]
nvcc will be used as CUDA compiler
Please specify the CUDA SDK version you want to use, e.g. 7.0. [Leave empty to default to CUDA 8.0]:
Please specify the location where CUDA  toolkit is installed. Refer to README.md for more details. [Default is /usr/local/cuda]:
Please specify which gcc should be used by nvcc as the host compiler. [Default is /usr/bin/gcc]:
Please specify the cuDNN version you want to use. [Leave empty to default to cuDNN 6.0]: 5.1.10
Please specify the location where cuDNN 5.1.10 library is installed. Refer to README.md for more details. [Default is /usr/local/cuda]:
Please specify a list of comma-separated Cuda compute capabilities you want to build with.
You can find the compute capability of your device at: https://developer.nvidia.com/cuda-gpus.
Please note that each additional compute capability significantly increases your build time and binary size.
[Default is: "3.5,5.2"]: 3.7
Do you wish to build TensorFlow with MPI support? [y/N]
MPI support will not be enabled for TensorFlow
Configuration finished
