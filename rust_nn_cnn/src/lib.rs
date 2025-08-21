#![recursion_limit = "512"]
use burn::module::Module;
use burn::nn::loss::{MseLoss, Reduction};
use burn::nn::{LinearConfig, conv::Conv2dConfig, pool::MaxPool2dConfig};
use burn::optim::Adam;
use burn::optim::Optimizer;
use burn::optim::adaptor::OptimizerAdaptor;
use burn::record::{FullPrecisionSettings, Recorder};
use burn::tensor::{Tensor, TensorData, backend::Backend};
use burn_autodiff::Autodiff;
use burn_wgpu::Wgpu;
use serde::{Deserialize, Serialize};

pub type DefaultGpuBackend = Wgpu<f32, i32, u32>;
pub type AutoDiffBackend = Autodiff<DefaultGpuBackend>;

// 1D CNN (generic inner) with user-specified FC sizes via new([input_len, ...fc_sizes])
#[derive(Module, Debug)]
pub struct NeuralNetworkInner<B: Backend> {
    in_channels: usize,
    input_len: usize,
    conv1: burn::nn::conv::Conv2d<B>, // kernel [1, k]
    conv2: burn::nn::conv::Conv2d<B>,
    linears: Vec<burn::nn::Linear<B>>, // built from sizes[1..]
}

impl<B: Backend> NeuralNetworkInner<B> {
    pub fn new(sizes: &[usize], device: &B::Device) -> Self {
        assert!(sizes.len() >= 2, "sizes: [input_len, fc1, ..., out]");
        let input_len = sizes[0];
        let in_channels = 1;
        // 1D conv via 2D conv with H=1
        let conv1 = Conv2dConfig::new([in_channels, 8], [1, 7]).init(device);
        let conv2 = Conv2dConfig::new([8, 16], [1, 5]).init(device);

        // Compute length after conv/pool: L1 = L-7+1; pool/2 -> L1p; L2 = L1p-5+1; pool/2 -> L2p
        let l1 = input_len.saturating_sub(7).saturating_add(1);
        let l1p = l1 / 2;
        let l2 = l1p.saturating_sub(5).saturating_add(1);
        let l2p = l2 / 2;
        let flatten_dim = 16 * l2p;

        // Build FC layers: first takes flatten_dim -> sizes[1], then chain sizes windows
        let mut linears = Vec::new();
        if sizes.len() >= 2 {
            linears.push(LinearConfig::new(flatten_dim, sizes[1]).init(device));
        }
        if sizes.len() > 2 {
            for w in sizes[1..].windows(2) {
                linears.push(LinearConfig::new(w[0], w[1]).init(device));
            }
        }

        Self {
            in_channels,
            input_len,
            conv1,
            conv2,
            linears,
        }
    }

    fn forward_inner(&self, x: Tensor<B, 2>, _device: &B::Device) -> Tensor<B, 2> {
        // x: [batch, L] -> reshape to [batch, C=1, H=1, W=L]
        let dims = x.dims();
        let batch = dims[0];
        let x = x.reshape([batch, self.in_channels, 1, self.input_len]);
        let x = self.conv1.forward(x);
        let x = burn::tensor::activation::relu(x);
        let x = MaxPool2dConfig::new([1, 2]).init().forward(x);
        let x = self.conv2.forward(x);
        let x = burn::tensor::activation::relu(x);
        let x = MaxPool2dConfig::new([1, 2]).init().forward(x);
        // Flatten [batch, 16, 1, L2p] -> [batch, 16*L2p]
        let w = x.dims()[3];
        let x = x.reshape([batch, 16 * w]);
        // FC chain
        let mut h = x;
        for (i, lin) in self.linears.iter().enumerate() {
            h = lin.forward(h);
            if i + 1 != self.linears.len() {
                h = burn::tensor::activation::relu(h);
            }
        }
        h
    }

    pub fn predict(&self, input: Vec<f32>, device: &B::Device) -> Vec<f32> {
        let x = Tensor::<B, 2>::from_data(TensorData::new(input, [1, self.input_len]), device);
        self.forward_inner(x, device)
            .into_data()
            .to_vec()
            .unwrap_or_default()
    }
}

impl NeuralNetworkInner<AutoDiffBackend> {
    pub fn train_batch(
        &mut self,
        inputs: &[Vec<f32>],
        targets: &[Vec<f32>],
        lr: f32,
        device: &<AutoDiffBackend as Backend>::Device,
    ) -> f32 {
        assert_eq!(inputs.len(), targets.len());
        let batch = inputs.len();
        let out_features = self
            .linears
            .last()
            .map(|l| {
                let [_din, dout] = l.weight.shape().dims::<2>();
                dout
            })
            .unwrap_or(0);

        let flat_inputs: Vec<f32> = inputs.iter().flat_map(|v| v.iter().copied()).collect();
        let flat_targets: Vec<f32> = targets.iter().flat_map(|v| v.iter().copied()).collect();

        let x = Tensor::<AutoDiffBackend, 2>::from_data(
            TensorData::new(flat_inputs, [batch, self.input_len]),
            device,
        );
        let y = Tensor::<AutoDiffBackend, 2>::from_data(
            TensorData::new(flat_targets, [batch, out_features]),
            device,
        );

        let logits = self.forward_inner(x, device);
        let loss = MseLoss::new().forward(logits, y, Reduction::Auto);

        let mut grads = loss.backward();
        let grads_params = burn::optim::GradientsParams::from_module(&mut grads, &*self);
        let mut optim = burn::optim::AdamConfig::new().init();
        use burn::optim::Optimizer;
        *self = optim.step(lr as f64, self.clone(), grads_params);

        let loss_vec: Vec<f32> = loss.into_data().to_vec().unwrap_or_default();
        *loss_vec.first().unwrap_or(&0.0)
    }
}

// Public simple wrapper that owns the backend device; user does not manage it
pub struct NeuralNetwork {
    device: <AutoDiffBackend as Backend>::Device,
    inner: NeuralNetworkInner<AutoDiffBackend>,
    lr: f32,
    optim: OptimizerAdaptor<Adam, NeuralNetworkInner<AutoDiffBackend>, AutoDiffBackend>,
    max_batch_size: usize,
    best_loss: Option<f32>,
    arch_sizes: Vec<usize>,
}

impl NeuralNetwork {
    pub fn new(sizes: &[usize], max_batch_size: usize) -> Self {
        let device = <AutoDiffBackend as Backend>::Device::default();
        let inner = NeuralNetworkInner::<AutoDiffBackend>::new(sizes, &device);
        let optim = burn::optim::AdamConfig::new().init();
        Self {
            device,
            inner,
            lr: 1e-3,
            optim,
            max_batch_size,
            best_loss: None,
            arch_sizes: sizes.to_vec(),
        }
    }

    pub fn with_learning_rate(mut self, lr: f32) -> Self {
        self.lr = lr;
        self
    }

    pub fn train_batch(&mut self, inputs: &[Vec<f32>], targets: &[Vec<f32>]) -> f32 {
        assert_eq!(inputs.len(), targets.len());
        let out_features = self
            .inner
            .linears
            .last()
            .map(|l| {
                let [_din, dout] = l.weight.shape().dims::<2>();
                dout
            })
            .unwrap_or(0);

        let mut idx = 0usize;
        let mut sum_loss = 0.0f32;
        let mut num_batches = 0usize;
        while idx < inputs.len() {
            let end = (idx + self.max_batch_size).min(inputs.len());

            let batch = end - idx;
            let flat_inputs: Vec<f32> = inputs[idx..end]
                .iter()
                .flat_map(|v| v.iter().copied())
                .collect();
            let flat_targets: Vec<f32> = targets[idx..end]
                .iter()
                .flat_map(|v| v.iter().copied())
                .collect();

            let x = Tensor::<AutoDiffBackend, 2>::from_data(
                TensorData::new(flat_inputs, [batch, self.inner.input_len]),
                &self.device,
            );
            let y = Tensor::<AutoDiffBackend, 2>::from_data(
                TensorData::new(flat_targets, [batch, out_features]),
                &self.device,
            );

            let logits = self.inner.forward_inner(x, &self.device);
            let loss = MseLoss::new().forward(logits, y, Reduction::Auto);

            let mut grads = loss.backward();
            let grads_params = burn::optim::GradientsParams::from_module(&mut grads, &self.inner);
            use burn::optim::Optimizer;
            self.inner = self
                .optim
                .step(self.lr as f64, self.inner.clone(), grads_params);

            let loss_vec: Vec<f32> = loss.into_data().to_vec().unwrap_or_default();
            let l = *loss_vec.first().unwrap_or(&0.0);
            sum_loss += l;
            num_batches += 1;
            idx = end;
        }
        sum_loss / num_batches as f32
    }

    pub fn predict(&self, input: Vec<f32>) -> Vec<f32> {
        self.inner.predict(input, &self.device)
    }

    pub fn save(&self, path: &str) -> Result<(), String> {
        use burn::record::NamedMpkFileRecorder;
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::default();
        let base = std::path::PathBuf::from(path);
        std::fs::create_dir_all(base.parent().unwrap_or_else(|| std::path::Path::new(".")))
            .map_err(|e| e.to_string())?;
        type ModelRecord =
            <NeuralNetworkInner<AutoDiffBackend> as burn::module::Module<AutoDiffBackend>>::Record;
        type OptimRecord = <OptimizerAdaptor<
            Adam,
            NeuralNetworkInner<AutoDiffBackend>,
            AutoDiffBackend,
        > as Optimizer<NeuralNetworkInner<AutoDiffBackend>, AutoDiffBackend>>::Record;
        let meta = ModelMeta {
            sizes: self.arch_sizes.clone(),
            lr: self.lr,
            max_batch_size: self.max_batch_size,
            best_loss: self.best_loss,
        };
        let bundle: (ModelRecord, OptimRecord, ModelMeta) = (
            self.inner.clone().into_record(),
            self.optim.to_record(),
            meta,
        );
        Recorder::record(&recorder, bundle, base).map_err(|e| format!("record error: {e}"))
    }

    pub fn load(path: &str) -> Result<Self, String> {
        let device = <AutoDiffBackend as Backend>::Device::default();
        let base = std::path::PathBuf::from(path);
        use burn::record::NamedMpkFileRecorder;
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::default();
        type ModelRecord =
            <NeuralNetworkInner<AutoDiffBackend> as burn::module::Module<AutoDiffBackend>>::Record;
        type OptimRecord = <OptimizerAdaptor<
            Adam,
            NeuralNetworkInner<AutoDiffBackend>,
            AutoDiffBackend,
        > as Optimizer<NeuralNetworkInner<AutoDiffBackend>, AutoDiffBackend>>::Record;
        let (model_rec, optim_rec, meta): (ModelRecord, OptimRecord, ModelMeta) =
            Recorder::load(&recorder, base, &device).map_err(|e| format!("load error: {e}"))?;
        let inner =
            NeuralNetworkInner::<AutoDiffBackend>::new(&meta.sizes, &device).load_record(model_rec);
        let mut optim = burn::optim::AdamConfig::new().init();
        optim = optim.load_record(optim_rec);
        Ok(Self {
            device,
            inner,
            lr: meta.lr,
            optim,
            max_batch_size: meta.max_batch_size,
            best_loss: meta.best_loss,
            arch_sizes: meta.sizes,
        })
    }

    pub fn best_loss(&self) -> Option<f32> {
        self.best_loss
    }

    pub fn try_save_checkpoint(&mut self, path: &str, current_loss: f32) -> Result<bool, String> {
        let is_better = self.best_loss.map(|b| current_loss < b).unwrap_or(true);
        if is_better {
            self.best_loss = Some(current_loss);
            self.save(path)?;
            Ok(true)
        } else {
            Ok(false)
        }
    }
}

#[derive(Serialize, Deserialize, burn::record::Record)]
struct ModelMeta {
    sizes: Vec<usize>,
    lr: f32,
    max_batch_size: usize,
    best_loss: Option<f32>,
}

// Convolutional neural network: conv -> relu -> pool -> conv -> relu -> pool -> flatten -> fc1 -> relu -> fc2
#[derive(Module, Debug)]
pub struct ConvNeuralNetwork<B: Backend> {
    in_channels: usize,
    conv1: burn::nn::conv::Conv2d<B>,
    conv2: burn::nn::conv::Conv2d<B>,
    fc1: burn::nn::Linear<B>,
    fc2: burn::nn::Linear<B>,
}

impl<B: Backend> ConvNeuralNetwork<B> {
    pub fn new(in_channels: usize, num_classes: usize, device: &B::Device) -> Self {
        // No padding, kernel 3x3, two conv blocks, MNIST 28x28 -> 26x26 -> pool 13x13 -> 11x11 -> pool 5x5
        let conv1 = Conv2dConfig::new([in_channels, 8], [3, 3]).init(device);
        let conv2 = Conv2dConfig::new([8, 16], [3, 3]).init(device);
        let fc1 = LinearConfig::new(16 * 5 * 5, 128).init(device);
        let fc2 = LinearConfig::new(128, num_classes).init(device);
        Self {
            in_channels,
            conv1,
            conv2,
            fc1,
            fc2,
        }
    }

    pub fn predict(&self, input: Vec<f32>, device: &B::Device) -> Vec<f32> {
        // Assume input is flattened [in_channels * 28 * 28]
        let bs = 1usize;
        let x = Tensor::<B, 4>::from_data(
            TensorData::new(input, [bs, self.in_channels, 28, 28]),
            device,
        );
        let logits = self.forward_inner(x);
        logits.into_data().to_vec().unwrap_or_default()
    }

    fn forward_inner(&self, x: Tensor<B, 4>) -> Tensor<B, 2> {
        let x = self.conv1.forward(x);
        let x = burn::tensor::activation::relu(x);
        let x = MaxPool2dConfig::new([2, 2]).init().forward(x);
        let x = self.conv2.forward(x);
        let x = burn::tensor::activation::relu(x);
        let x = MaxPool2dConfig::new([2, 2]).init().forward(x);
        // Flatten to [batch, 16*5*5]
        let batch = x.dims()[0];
        let x = x.reshape([batch, 16 * 5 * 5]);
        let x = self.fc1.forward(x);
        self.fc2.forward(x)
    }
}

impl ConvNeuralNetwork<AutoDiffBackend> {
    pub fn train_batch(
        &mut self,
        inputs: &[Vec<f32>],
        targets: &[Vec<f32>],
        lr: f32,
        device: &<AutoDiffBackend as Backend>::Device,
    ) -> f32 {
        assert_eq!(inputs.len(), targets.len());
        let batch = inputs.len();
        let out_features = self.fc2.weight.shape().dims::<2>()[1];

        let flat_inputs: Vec<f32> = inputs.iter().flat_map(|v| v.iter().copied()).collect();
        let flat_targets: Vec<f32> = targets.iter().flat_map(|v| v.iter().copied()).collect();

        let x = Tensor::<AutoDiffBackend, 4>::from_data(
            TensorData::new(flat_inputs, [batch, self.in_channels, 28, 28]),
            device,
        );
        let y = Tensor::<AutoDiffBackend, 2>::from_data(
            TensorData::new(flat_targets, [batch, out_features]),
            device,
        );

        let logits = self.forward_inner(x);
        let loss = MseLoss::new().forward(logits, y, Reduction::Auto);

        let mut grads = loss.backward();
        let grads_params = burn::optim::GradientsParams::from_module(&mut grads, &*self);
        let mut optim = burn::optim::AdamConfig::new().init();
        use burn::optim::Optimizer;
        *self = optim.step(lr as f64, self.clone(), grads_params);

        let loss_vec: Vec<f32> = loss.into_data().to_vec().unwrap_or_default();
        *loss_vec.first().unwrap_or(&0.0)
    }
}
