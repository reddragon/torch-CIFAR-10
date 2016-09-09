require 'nn'
require 'paths'

if (not paths.filep("cifar10torchsmall.zip")) then
  print("Getting the small CIFAR-10 dataset.")
  os.execute('wget -c https://s3.amazonaws.com/torch7/data/cifar10torchsmall.zip')
  os.execute('unzip cifar10torchsmall.zip')
end
