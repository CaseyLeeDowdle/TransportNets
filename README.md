# TransportNets

TransportNets is complicated programming.

## Installation

The recommended way to run TransportNets is to first install a docker container with TensorFlow 2.x. Then, bind mount to a local directory.

### TensorFlow with Docker

Follow the instructions here to install TensorFlow with Docker: https://www.tensorflow.org/install/docker

Run the container with:

`docker run -it -v ~/TRANSPORTNETS:/local -p 8888:8888 tensorflow/tensorflow:2.X.X-jupyter bash`

`TRANSPORTNETS` is the directory in which TransportNets is located.

Then `cd local/` to access the local `TRANSPORTNETS` folder. From there, you can run a jupyter notebook using `jupyter notebook --ip=0.0.0.0` (replacing the ip with the one you are using).
