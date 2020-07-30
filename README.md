# TransportNets

TransportNets is complicated.

## Installation

The recommended way to run TransportNets is to first install a docker container with TensorFlow 2.x. Then, bind mount to a local directory.

### TensorFlow with Docker

1. Follow the instructions here to install TensorFlow with Docker: https://www.tensorflow.org/install/docker.  Note that there are many different docker images with and without GPU support.   We recommend using one of the jupyter images, such as `tensorflow/tensorflow:latest-jupyter` or `tensorflow/tensorflow:latest-gpu-jupyter`.   Many of the TransportNets examples are provided in jupyter notebooks.

2. Run a docker container using the tensorflow image and share the TransportNets folder with the container.  For example, run 
```docker run -it -v ~/<transportnets_folder>:/local -p 8888:8888 tensorflow/tensorflow:latest-jupyter bash```
where `<transportnets_folder>` is the directory where this repository is located on your host machine.

3. Then `cd ~/<transportnets_folder>/` to access the contents of this repository. From there, you can run a jupyter notebook using `jupyter notebook --ip=*`, which will be accessible on port 8888 of your host machine.
