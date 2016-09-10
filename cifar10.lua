require 'image'
require 'nn'
require 'paths'

if (not paths.filep("cifar10torchsmall.zip")) then
  print("Getting the small CIFAR-10 dataset.")
  os.execute('wget -c https://s3.amazonaws.com/torch7/data/cifar10torchsmall.zip')
  os.execute('unzip cifar10torchsmall.zip')
end

trainset = torch.load('cifar10-train.t7')
testset = torch.load('cifar10-test.t7')

print(trainset)
print(#trainset.data)

-- image.display(trainset.data[100])

setmetatable(trainset,
  {
    __index = function(t, i)
                return {t.data[i], t.label[i]}
              end
  }
);

function trainset:size()
    return self.data:size(1)
end

print(trainset:size())

redChannel = trainset.data[{ {}, {1}, {}, {}  }]
print(#redChannel)

mean = {} -- store the mean, to normalize the test set in the future
stdv  = {} -- store the standard-deviation for the future

if trainset then
  print 'It is set fine'
  print(#trainset.data[{ {}, {i}, {}, {}  }])
else
  print 'No, it is not set'
end

trainset.data = trainset.data:double()
for i=1,3 do -- over each image channel
    mean[i] = torch.DoubleTensor(trainset.data[{ {}, {i}, {}, {}  }]):mean() -- mean estimation
    print('Channel ' .. i .. ', Mean: ' .. mean[i])
    trainset.data[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction

    stdv[i] = torch.DoubleTensor(trainset.data[{ {}, {i}, {}, {}  }]):std() -- std estimation
    print('Channel ' .. i .. ', Standard Deviation: ' .. stdv[i])
    trainset.data[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
end

net = nn.Sequential()
net:add(nn.SpatialConvolution(3, 6, 5, 5)) -- 3 input image channels, 6 output channels, 5x5 convolution kernel
net:add(nn.ReLU())                       -- non-linearity
net:add(nn.SpatialMaxPooling(2,2,2,2))     -- A max-pooling operation that looks at 2x2 windows and finds the max.
net:add(nn.SpatialConvolution(6, 16, 5, 5))
net:add(nn.ReLU())                       -- non-linearity
net:add(nn.SpatialMaxPooling(2,2,2,2))
net:add(nn.View(16*5*5))                    -- reshapes from a 3D tensor of 16x5x5 into 1D tensor of 16*5*5
net:add(nn.Linear(16*5*5, 120))             -- fully connected layer (matrix multiplication between input and weights)
net:add(nn.ReLU())                       -- non-linearity
net:add(nn.Linear(120, 84))
net:add(nn.ReLU())                       -- non-linearity
net:add(nn.Linear(84, 10))                   -- 10 is the number of outputs of the network (in this case, 10 digits)
net:add(nn.LogSoftMax())                     -- converts the output to a log-probability. Useful for classification problems

criterion = nn.ClassNLLCriterion()
trainer = nn.StochasticGradient(net, criterion)
trainer.learningRate = 0.001
trainer.maxIteration = 5 -- just do 5 epochs of training.

trainer:train(trainset)

testset.data = testset.data:double()   -- convert from Byte tensor to Double tensor
for i=1,3 do -- over each image channel
    testset.data[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction
    testset.data[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
end

correct = 0
for i=1,10000 do
    local groundtruth = testset.label[i]
    local prediction = net:forward(testset.data[i])
    local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
    if groundtruth == indices[1] then
        correct = correct + 1
    end
end
print(correct, 100*correct/10000 .. ' % ')
