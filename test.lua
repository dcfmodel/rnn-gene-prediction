
--[[

This file samples characters from a trained model

Code is based on implementation in 
https://github.com/oxford-cs-ml-2015/practical6

]]--

require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'

require 'util.OneHot'
require 'util.misc'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Sample from a character-level language model')
cmd:text()
cmd:text('Options')
-- required:
cmd:argument('-model','model checkpoint to use for sampling')
-- optional parameters
cmd:option('-seed',123,'random number generator\'s seed')
cmd:option('-sample',1,' 0 to use max at each timestep, 1 to sample at each timestep')
cmd:option('-primetext',"",'used as a prompt to "seed" the state of the LSTM using a given sequence, before we sample.')
cmd:option('-length',2000,'number of characters to sample')
cmd:option('-temperature',1,'temperature of sampling')
cmd:option('-gpuid',-1,'which gpu to use. -1 = use CPU')
cmd:option('-opencl',0,'use OpenCL (instead of CUDA)')
cmd:option('-verbose',1,'set to 0 to ONLY print the sampled text, no diagnostics')
cmd:text()

-- parse input params
opt = cmd:parse(arg)

-- gated print: simple utility function wrapping a print
function gprint(str)
    if opt.verbose == 1 then print(str) end
end

-- check that cunn/cutorch are installed if user wants to use the GPU
if opt.gpuid >= 0 and opt.opencl == 0 then
    local ok, cunn = pcall(require, 'cunn')
    local ok2, cutorch = pcall(require, 'cutorch')
    if not ok then gprint('package cunn not found!') end
    if not ok2 then gprint('package cutorch not found!') end
    if ok and ok2 then
        gprint('using CUDA on GPU ' .. opt.gpuid .. '...')
        cutorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
        cutorch.manualSeed(opt.seed)
    else
        gprint('Falling back on CPU mode')
        opt.gpuid = -1 -- overwrite user setting
    end
end

-- check that clnn/cltorch are installed if user wants to use OpenCL
if opt.gpuid >= 0 and opt.opencl == 1 then
    local ok, cunn = pcall(require, 'clnn')
    local ok2, cutorch = pcall(require, 'cltorch')
    if not ok then print('package clnn not found!') end
    if not ok2 then print('package cltorch not found!') end
    if ok and ok2 then
        print('using OpenCL on GPU ' .. opt.gpuid .. '...')
        cltorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
        torch.manualSeed(opt.seed)
    else
        gprint('Falling back on CPU mode')
        opt.gpuid = -1 -- overwrite user setting
    end
end

torch.manualSeed(opt.seed)

-- load the model checkpoint
if not lfs.attributes(opt.model, 'mode') then
    gprint('Error: File ' .. opt.model .. ' does not exist. Are you sure you didn\'t forget to prepend cv/ ?')
end
checkpoint = torch.load(opt.model)
protos = checkpoint.protos
protos.rnn:evaluate() -- put in eval mode so that dropout works properly

-- initialize the vocabulary (and its inverted version)
local vocab = checkpoint.vocab
local ivocab = {}
for c,i in pairs(vocab) do ivocab[i] = c end

-- initialize the rnn state to all zeros
gprint('creating an ' .. checkpoint.opt.model .. '...')
local current_state
current_state = {}
for L = 1,checkpoint.opt.num_layers do
    -- c and h for all layers
    local h_init = torch.zeros(1, checkpoint.opt.rnn_size)
    if opt.gpuid >= 0 and opt.opencl == 0 then h_init = h_init:cuda() end
    if opt.gpuid >= 0 and opt.opencl == 1 then h_init = h_init:cl() end
    table.insert(current_state, h_init:clone())
    if checkpoint.opt.model == 'lstm' then
        table.insert(current_state, h_init:clone())
    end
end
state_size = #current_state

-- set input/output paths
local input_file = path.join(opt.data_dir, 'input.txt')
local label_file = path.join(opt.data_dir, 'labels.txt')
local input_tensorFile = path.join(opt.data_dir, 'data.t7')
local output_file = path.join(opt.out_dir, 'predictions.txt')

-- fetch file attributes to determine if we need to rerun preprocessing
local run_prepro = false
if not (path.exists(input_tensorFile)) then
    -- prepro files do not exist, generate them
    print('data.t7 does not exist. Running preprocessing...')
    run_prepro = true
else
    -- check if the input file was modified since last time we 
    -- ran the prepro. if so, we have to rerun the preprocessing
    local input_attr = lfs.attributes(input_file)
    local tensor_attr = lfs.attributes(input_tensorFile)
    if input_attr.modification > tensor_attr.modification then
        print('data.t7 detected as stale. Re-running preprocessing...')
        run_prepro = true
    end
end
if run_prepro then
    -- construct a tensor with all the data by making use of loader class function: text_to_tensor
    print('one-time setup: preprocessing input text file ' .. input_file .. '...')
    CharSplitLMMinibatchLoader.text_to_tensor(input_file, nil, input_tensorFile, label_file, vocab)
end

--[[
STOPPED DEVELOPING HERE... REALIZED 'train.lua' COULD BE USED TO TEST EVEN WHEN
USING 2 DATASETS.  JUST LOAD CHECKPOINT FROM DATASET 1, AND SPECIFY TEST FRACTION
FOR DATASET 2
]]

-- load the input tensor
local input = torch.load(input_tensorFile)
-- put into left, top, width, height format
input[{{3},{}}] = input[{{3},{}}] - input[{{1},{}}] + 1
input[{{4},{}}] = input[{{4},{}}] - input[{{4},{}}] + 1

print('writing predictions to ' .. output_txtFile)
-- open output file
file = torch.DiskFile(output_txtFile, 'w')
file:writeString(tostring(opt.frames)..' \n')

-- loop through input bounding boxes to make predictions at each time step
local prediction	-- localize prediction to main chunk
for f = 1, seq_length - opt.frames do
	
	-- do forward pass with input for frame i
	prediction = forwardPass(input[{{},{f}}]:t())

	-- if predicting more than 1 frame propagate prediction through the next 'frames' time steps
	if opt.frames > 1 then
		for i = 1, opt.frames-1 do prediction = forwardPass(prediction) end
	end

	-- write the prediction to txt file
	s = tostring(prediction)
	last = s:find('\n') - 1
	for element in s:sub(2,last):gmatch'%S+' do file:writeString(element..' ') end
	file:writeString('\n')

end

file:close()

