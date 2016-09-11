require 'image'
require 'nn'
require 'paths'
require 'clnn'
require 'cltorch'

function setupData()
  if (not paths.filep("cifar10torchsmall.zip")) then
    print("Getting the small CIFAR-10 dataset.")
    os.execute('wget -c https://s3.amazonaws.com/torch7/data/cifar10torchsmall.zip')
    os.execute('unzip cifar10torchsmall.zip')
  end
  local trainset = torch.load('cifar10-train.t7')
  local testset = torch.load('cifar10-test.t7')
  trainset.data = trainset.data:double()
  testset.data = testset.data:double()

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
  return trainset, testset
end

function normalizeTrainSet(trainset)
  local mean = {}
  local stddev = {}

  for i=1,3 do -- over each image channel
      mean[i] = torch.DoubleTensor(trainset.data[{ {}, {i}, {}, {}  }]):mean()
      trainset.data[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction

      stddev[i] = torch.DoubleTensor(trainset.data[{ {}, {i}, {}, {}  }]):std()
      trainset.data[{ {}, {i}, {}, {}  }]:div(stddev[i]) -- std scaling
  end
  return trainset, mean, stddev
end

function normalizeTestSet(testset, mean, stddev)
  for i=1,3 do -- over each image channel
      testset.data[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction
      testset.data[{ {}, {i}, {}, {}  }]:div(stddev[i]) -- std scaling
  end
  return testset
end

function setupNet()
  local net = nn.Sequential()
  net:add(nn.SpatialConvolution(3, 6, 5, 5)) -- 3 input image channels, 6 output channels, 5x5 convolution kernel
  net:add(nn.ReLU())                       -- non-linearity
  net:add(nn.SpatialMaxPooling(2,2,2,2))     -- A max-pooling operation that looks at 2x2 windows and finds the max.
  net:add(nn.SpatialConvolution(6, 16, 5, 5))
  net:add(nn.ReLU())                       -- non-linearity
  net:add(nn.SpatialMaxPooling(2,2,2,2))
  net:add(nn.View(16*5*5))                    -- reshapes from a 3D tensor of 16x5x5 into 1D tensor of 16*5*5
  net:add(nn.Linear(16*5*5, 120))             -- fully connected layer (matrix multiplication between input and weights)
  net:add(nn.ReLU())                       -- non-linearity
  net:add(nn.Linear(120, 10))
  -- net:add(nn.Linear(120, 84))
  -- net:add(nn.ReLU())                       -- non-linearity
  -- net:add(nn.Linear(84, 10))                   -- 10 is the number of outputs of the network (in this case, 10 digits)
  net:add(nn.LogSoftMax())                     -- converts the output to a log-probability. Useful for classification problems
  return net
end

function train(trainset)
  local net = setupNet()
  local criterion = nn.ClassNLLCriterion()
  local trainer = nn.StochasticGradient(net, criterion)
  trainer.learningRate = 0.001
  trainer.maxIteration = 1 -- just do 5 epochs of training.
  trainer:train(trainset)
  return net
end

function main()
  print('Preparing the train & test data.')
  trainset, testset = setupData()
  trainset, mean, stddev = normalizeTrainSet(trainset)
  print('Starting the training.')
  net = train(trainset)

  print('Starting the test Evaluation.')
  testset = normalizeTestSet(testset, mean, stddev)
  correct = 0
  for i=1,10000 do
    local groundtruth = testset.label[i]
    local prediction = net:forward(testset.data[i])
    local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
    if groundtruth == indices[1] then
        correct = correct + 1
    end
  end
  print('Test Accuracy:', 100*correct/10000 .. '%')
end

main()
