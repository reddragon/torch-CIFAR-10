# Torch CIFAR-10 Demo

Inspired ~~Mostly blatantly copied~~ from @soumith's [Deep Learning Intro with Torch](https://github.com/soumith/cvpr2015/blob/master/Deep%20Learning%20with%20Torch.ipynb). Runs on a subset of CIFAR-10 (10k training examples, which I believe is what is causing a bottleneck in the test accuracy that I get).

### My contributions:
* I added the ability to parameterize hyper-params (learning rate, epochs/iterations).
* Added support for running on OpenCL & CUDA.
* Changed the network to add more Convolutional layers.
* Modularized the code.

### Coming Up
* Use of `optim` along with L-BFGS/Adam.
* Using the entire CIFAR-10 dataset, instead of the 10k train + 10k test currently.

## Dependencies
* `nn` for Neural Nets, obviously.
* `clnn`, `cltorch` if using OpenCL
* `cunn`, `cutorch` if using CUDA


## Demo
* If you want to run the demo on your CPU, try `th cifar10.lua`.
* For GPUs, figure out which & how many GPUs you have via `lspci | grep NVIDIA` or `lspci | grep AMD`.
* `th cifar10.lua -gpu 0 -backend cunn` - Will run the demo on your NVIDIA GPU number 1 (0-indexed).
* `th cifar10.lua -gpu 0 -backend clnn` - Will run the demo on your AMD GPU number 1 (0-indexed).
* To experiment with iterations & learning rate, try something like this `th cifar10.lua -gpu 1 -backend cunn -iters 100 -lr 0.0005`.
* So far, I am able to see an ~~49.2~~ 57% accuracy on test.

