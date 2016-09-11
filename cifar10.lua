require 'image'
require 'nn'
require 'paths'

local cmd = torch.CmdLine()
cmd:option('-gpu', -1, 'Zero-indexed ID of the GPU to use; Use -1 for CPU mode.')
cmd:option('-backend', 'nn', 'nn for CPU, cunn for CUDA, clnn for OpenCL.')

local function setupData()
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

local function normalizeTrainSet(trainset)
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

local function normalizeTestSet(testset, mean, stddev)
  for i=1,3 do -- over each image channel
      testset.data[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction
      testset.data[{ {}, {i}, {}, {}  }]:div(stddev[i]) -- std scaling
  end
  return testset
end

local function setupNet()
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

local function train(trainset, params)
  local net = setupNet()
  local criterion = nn.ClassNLLCriterion()

  if params.gpu >= 0 then
    if params.backend ~= 'clnn' then
      -- CUDA backend, move everything to CUDA.
      trainset.data = trainset.data:cuda()
      trainset.label = trainset.label:cuda()
      testset.data = testset.data:cuda()
      testset.label = testset.label:cuda()
      net = net:cuda()
      criterion = criterion:cuda()
      print('Moving the training to the CUDA backend.')
    else
      -- OpenCL backend, move everything to OpenCL.
      trainset.data = trainset.data:cl()
      trainset.label = trainset.label:cl()
      testset.data = testset.data:cl()
      testset.label = testset.label:cl()
      net = net:cl()
      criterion = criterion:cl()
      print('Moving the training to the OpenCL backend.')
    end
  end

  local trainer = nn.StochasticGradient(net, criterion)
  trainer.learningRate = 0.001
  trainer.maxIteration = 5
  trainer:train(trainset)
  return net
end

local function main(params)
  local usingOpenCl = false
  if params.gpu >= 0 then
    if params.backend ~= 'clnn' then
      require 'cunn'
      require 'cutorch'
      cutorch.setDevice(params.gpu + 1)
    else
      require 'clnn'
      require 'cltorch'
      cltorch.setDevice(params.gpu + 1)
      usingOpenCl = true
    end
  end

  print('Preparing the train & test data.')
  trainset, testset = setupData()
  trainset, mean, stddev = normalizeTrainSet(trainset)
  print('Starting the training.')
  net = train(trainset, params)

  print('Starting the test Evaluation.')
  testset = normalizeTestSet(testset, mean, stddev)
  correct = 0
  for i=1,10000 do
    local groundtruth = testset.label[i]
    local prediction = net:forward(testset.data[i])

    if usingOpenCl then
      -- OpenCL doesnt support torch.sort on torch.CLTensor, yet.
      -- Moving the predictions to CPU shouldn't be a big problem.
      prediction = prediction:float()
    end

    local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
    if groundtruth == indices[1] then
        correct = correct + 1
    end
  end
  print('Test Accuracy:', 100*correct/10000 .. '%')
end

local params = cmd:parse(arg)
main(params)
print('Collecting garbage before exiting.')
collectgarbage()
