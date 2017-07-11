%% set opts for training (see edgesTrain.m)
opts = edgesTrain();                % default options (good settings)
opts.modelDir = 'models/';          % model will be in models/forest
opts.modelFnm = 'modelBsds';        % model name
opts.nPos = 5e5; opts.nNeg = 5e5;   % decrease to speedup training
opts.useParfor = 0;                 % parallelize if sufficient memory

%% train edge detector (~20m/8Gb per tree, proportional to nPos/nNeg)
model = edgesTrain(opts); % will load model if already trained

%% set detection parameters (can set after training)
model.opts.multiscale = 0;          % for top accuracy set multiscale = 1
model.opts.sharpen = 0;             % for top speed set sharpen = 0
model.opts.nTreesEval = 1;          % for top speed set nTreesEval = 1
model.opts.nThreads = 4;            % max number threads for evaluation
model.opts.nms = 0;                 % set to true to enable nms

I = imread('peppers.png');
I = im2double(I);

%% Edge-preserving smoothing example
sigma_s = 20;
sigma_r = 0.04;
tic
[F, E] = SemanticFilter(I, sigma_s, sigma_r, model, 10);
toc
figure(1);imshow(I);
figure(3);imshow(E);
figure(2);imshow(F);
