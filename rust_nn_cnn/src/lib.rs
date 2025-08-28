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
use burn_ndarray::NdArray;
use burn_wgpu::Wgpu;
use serde::{Deserialize, Serialize};
use wgpu::{Backends, Instance, InstanceDescriptor, PowerPreference, RequestAdapterOptions};

pub type DefaultGpuBackend = Wgpu<f32, i32, u32>;
pub type AutoDiffBackend = Autodiff<DefaultGpuBackend>; // GPU autodiff
pub type DefaultCpuBackend = NdArray<f32, i32>;
pub type AutoDiffCpuBackend = Autodiff<DefaultCpuBackend>; // CPU autodiff

/// GPU backend preference for device initialization.
/// - `Default`: uses `wgpu`'s default selection to play nice with other GPU users.
/// - `Max`: forces Vulkan backend for maximum performance on systems with Vulkan.
#[derive(Clone, Copy, Debug)]
pub enum GpuBackend {
    Default,
    Max,
}

fn apply_gpu_backend_env(profile: GpuBackend) {
    match profile {
        GpuBackend::Default => {
            // Prefer discrete GPUs and choose non-Vulkan backends explicitly per-OS.
            // Hint high-performance adapter selection universally.
            unsafe {
                std::env::set_var("WGPU_POWER_PREF", "high");
            }
            // We avoid forcing DRI_PRIME to prevent driver warnings on single-GPU systems.
            // Backend selection: Linux: OpenGL; macOS: Metal; Windows: DX12.
            // Linux: OpenGL; macOS: Metal; Windows: DX12.
            #[cfg(target_os = "linux")]
            unsafe {
                std::env::set_var("WGPU_BACKENDS", "GL");
            }
            #[cfg(target_os = "macos")]
            unsafe {
                std::env::set_var("WGPU_BACKENDS", "METAL");
            }
            #[cfg(target_os = "windows")]
            unsafe {
                std::env::set_var("WGPU_BACKENDS", "DX12");
            }
            #[cfg(all(
                not(target_os = "linux"),
                not(target_os = "macos"),
                not(target_os = "windows")
            ))]
            unsafe {
                // Fallback: let wgpu choose backend; still prefer high-performance.
                std::env::remove_var("WGPU_BACKENDS");
            }
        }
        GpuBackend::Max => {
            // Prefer discrete GPUs and force Vulkan backend when available.
            unsafe {
                std::env::set_var("WGPU_POWER_PREF", "high");
            }
            // Avoid forcing DRI_PRIME here as well.
            unsafe {
                std::env::set_var("WGPU_BACKENDS", "VULKAN");
            }
        }
    }
}

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

    pub fn predict(&self, inputs: &[Vec<f32>], device: &B::Device) -> Vec<Vec<f32>> {
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
        let x = Tensor::<B, 2>::from_data(
            TensorData::new(flat_inputs, [batch, self.input_len]),
            device,
        );
        let logits = self.forward_inner(x, device);
        let vec = logits.into_data().to_vec().unwrap_or_default();
        if out_features == 0 {
            return vec![Vec::new(); batch];
        }
        let mut out: Vec<Vec<f32>> = Vec::with_capacity(batch);
        for chunk in vec.chunks(out_features) {
            out.push(chunk.to_vec());
        }
        out
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

// GPU-specific wrapper that owns the device; user does not manage it
pub struct NeuralNetworkGpu {
    device: <AutoDiffBackend as Backend>::Device,
    inner: NeuralNetworkInner<AutoDiffBackend>,
    lr: f32,
    optim: OptimizerAdaptor<Adam, NeuralNetworkInner<AutoDiffBackend>, AutoDiffBackend>,
    max_batch_size: usize,
    best_loss: Option<f32>,
    arch_sizes: Vec<usize>,
}

impl NeuralNetworkGpu {
    /// Create a new network with the specified architecture, batch size, learning rate, and GPU backend profile.
    pub fn new(sizes: &[usize], max_batch_size: usize, lr: f32, backend: GpuBackend) -> Self {
        apply_gpu_backend_env(backend);
        let device = <AutoDiffBackend as Backend>::Device::default();
        let inner = NeuralNetworkInner::<AutoDiffBackend>::new(sizes, &device);
        let optim = burn::optim::AdamConfig::new().init();
        Self {
            device,
            inner,
            lr,
            optim,
            max_batch_size,
            best_loss: None,
            arch_sizes: sizes.to_vec(),
        }
    }

    /// Deprecated: learning rate is now configured via `new`.
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

    pub fn predict(&self, inputs: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let mut outputs: Vec<Vec<f32>> = Vec::with_capacity(inputs.len());
        let mut idx = 0usize;
        while idx < inputs.len() {
            let end = (idx + self.max_batch_size).min(inputs.len());
            let batch_out = self.inner.predict(&inputs[idx..end], &self.device);
            outputs.extend(batch_out);
            idx = end;
        }
        outputs
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

    /// Load a model from a file base path (without `.mpk`) choosing a GPU backend profile.
    pub fn load_with_backend(path: &str, backend: GpuBackend) -> Result<Self, String> {
        apply_gpu_backend_env(backend);
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

    /// Load a model from binary bytes with a chosen GPU backend profile.
    pub fn load_from_bytes_with_backend(bytes: &[u8], backend: GpuBackend) -> Result<Self, String> {
        apply_gpu_backend_env(backend);
        let device = <AutoDiffBackend as Backend>::Device::default();
        use burn::record::NamedMpkFileRecorder;
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::default();

        // Write bytes to a temporary `.mpk` file and load via the recorder API.
        let tmp_dir = tempfile::tempdir().map_err(|e| e.to_string())?;
        let base = tmp_dir.path().join("bundle");
        let mpk_path = base.with_extension("mpk");
        std::fs::write(&mpk_path, bytes).map_err(|e| e.to_string())?;

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

// CPU-specific wrapper that owns the device; used for fallback or explicit CPU runs
pub struct NeuralNetworkCpu {
    device: <AutoDiffCpuBackend as Backend>::Device,
    inner: NeuralNetworkInner<AutoDiffCpuBackend>,
    lr: f32,
    optim: OptimizerAdaptor<Adam, NeuralNetworkInner<AutoDiffCpuBackend>, AutoDiffCpuBackend>,
    max_batch_size: usize,
    best_loss: Option<f32>,
    arch_sizes: Vec<usize>,
}

impl NeuralNetworkCpu {
    pub fn new(sizes: &[usize], max_batch_size: usize, lr: f32) -> Self {
        let device = <AutoDiffCpuBackend as Backend>::Device::default();
        let inner = NeuralNetworkInner::<AutoDiffCpuBackend>::new(sizes, &device);
        let optim = burn::optim::AdamConfig::new().init();
        Self {
            device,
            inner,
            lr,
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

            let x = Tensor::<AutoDiffCpuBackend, 2>::from_data(
                TensorData::new(flat_inputs, [batch, self.inner.input_len]),
                &self.device,
            );
            let y = Tensor::<AutoDiffCpuBackend, 2>::from_data(
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

    pub fn predict(&self, inputs: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let mut outputs: Vec<Vec<f32>> = Vec::with_capacity(inputs.len());
        let mut idx = 0usize;
        while idx < inputs.len() {
            let end = (idx + self.max_batch_size).min(inputs.len());
            let batch_out = self.inner.predict(&inputs[idx..end], &self.device);
            outputs.extend(batch_out);
            idx = end;
        }
        outputs
    }

    pub fn save(&self, path: &str) -> Result<(), String> {
        use burn::record::NamedMpkFileRecorder;
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::default();
        let base = std::path::PathBuf::from(path);
        std::fs::create_dir_all(base.parent().unwrap_or_else(|| std::path::Path::new(".")))
            .map_err(|e| e.to_string())?;
        type ModelRecord = <NeuralNetworkInner<AutoDiffCpuBackend> as burn::module::Module<
            AutoDiffCpuBackend,
        >>::Record;
        type OptimRecord = <OptimizerAdaptor<
            Adam,
            NeuralNetworkInner<AutoDiffCpuBackend>,
            AutoDiffCpuBackend,
        > as Optimizer<NeuralNetworkInner<AutoDiffCpuBackend>, AutoDiffCpuBackend>>::Record;
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

    pub fn load_from_bytes_with_backend(bytes: &[u8]) -> Result<Self, String> {
        let device = <AutoDiffCpuBackend as Backend>::Device::default();
        use burn::record::NamedMpkFileRecorder;
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::default();

        let tmp_dir = tempfile::tempdir().map_err(|e| e.to_string())?;
        let base = tmp_dir.path().join("bundle");
        let mpk_path = base.with_extension("mpk");
        std::fs::write(&mpk_path, bytes).map_err(|e| e.to_string())?;

        type ModelRecord = <NeuralNetworkInner<AutoDiffCpuBackend> as burn::module::Module<
            AutoDiffCpuBackend,
        >>::Record;
        type OptimRecord = <OptimizerAdaptor<
            Adam,
            NeuralNetworkInner<AutoDiffCpuBackend>,
            AutoDiffCpuBackend,
        > as Optimizer<NeuralNetworkInner<AutoDiffCpuBackend>, AutoDiffCpuBackend>>::Record;

        let (model_rec, optim_rec, meta): (ModelRecord, OptimRecord, ModelMeta) =
            Recorder::load(&recorder, base, &device).map_err(|e| format!("load error: {e}"))?;

        let inner = NeuralNetworkInner::<AutoDiffCpuBackend>::new(&meta.sizes, &device)
            .load_record(model_rec);
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
}

/// Public runtime-dispatch wrapper; may hold a GPU or CPU model.
pub enum NeuralNetwork {
    Gpu(NeuralNetworkGpu),
    Cpu(NeuralNetworkCpu),
}

fn try_init_gpu_device(profile: GpuBackend) -> Option<<AutoDiffBackend as Backend>::Device> {
    apply_gpu_backend_env(profile);

    // Proactively probe for a suitable adapter; if none, fall back to CPU.
    let backends = match profile {
        GpuBackend::Default => {
            #[cfg(target_os = "linux")]
            {
                Backends::GL
            }
            #[cfg(target_os = "macos")]
            {
                Backends::METAL
            }
            #[cfg(target_os = "windows")]
            {
                Backends::DX12
            }
            #[cfg(all(
                not(target_os = "linux"),
                not(target_os = "macos"),
                not(target_os = "windows")
            ))]
            {
                Backends::all()
            }
        }
        GpuBackend::Max => Backends::VULKAN,
    };
    let instance = Instance::new(InstanceDescriptor {
        backends,
        ..Default::default()
    });
    let opts = RequestAdapterOptions {
        power_preference: PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    };
    let adapter = pollster::block_on(instance.request_adapter(&opts));
    adapter.as_ref()?;

    // If wgpu can find an adapter, attempt to init burn's device. Catch panics and fall back.
    let res = std::panic::catch_unwind(<AutoDiffBackend as Backend>::Device::default);
    res.ok()
}

impl NeuralNetwork {
    pub fn new(sizes: &[usize], max_batch_size: usize, lr: f32, backend: GpuBackend) -> Self {
        if let Some(device) = try_init_gpu_device(backend) {
            // Build GPU variant using the pre-initialized device
            let inner = NeuralNetworkInner::<AutoDiffBackend>::new(sizes, &device);
            let optim = burn::optim::AdamConfig::new().init();
            NeuralNetwork::Gpu(NeuralNetworkGpu {
                device,
                inner,
                lr,
                optim,
                max_batch_size,
                best_loss: None,
                arch_sizes: sizes.to_vec(),
            })
        } else {
            // Fallback to CPU
            NeuralNetwork::Cpu(NeuralNetworkCpu::new(sizes, max_batch_size, lr))
        }
    }

    pub fn with_learning_rate(self, lr: f32) -> Self {
        match self {
            NeuralNetwork::Gpu(mut g) => {
                g.lr = lr;
                NeuralNetwork::Gpu(g)
            }
            NeuralNetwork::Cpu(c) => NeuralNetwork::Cpu(c.with_learning_rate(lr)),
        }
    }

    pub fn train_batch(&mut self, inputs: &[Vec<f32>], targets: &[Vec<f32>]) -> f32 {
        match self {
            NeuralNetwork::Gpu(g) => g.train_batch(inputs, targets),
            NeuralNetwork::Cpu(c) => c.train_batch(inputs, targets),
        }
    }

    pub fn predict(&self, inputs: &[Vec<f32>]) -> Vec<Vec<f32>> {
        match self {
            NeuralNetwork::Gpu(g) => g.predict(inputs),
            NeuralNetwork::Cpu(c) => c.predict(inputs),
        }
    }

    pub fn save(&self, path: &str) -> Result<(), String> {
        match self {
            NeuralNetwork::Gpu(g) => g.save(path),
            NeuralNetwork::Cpu(c) => c.save(path),
        }
    }

    pub fn load(path: &str) -> Result<Self, String> {
        // Try GPU Default, fall back to CPU if GPU init fails or load fails
        if try_init_gpu_device(GpuBackend::Default).is_some() {
            // Use GPU loader
            if let Ok(g) = NeuralNetworkGpu::load_with_backend(path, GpuBackend::Default) {
                return Ok(NeuralNetwork::Gpu(g));
            }
        }
        // CPU load
        let device = <AutoDiffCpuBackend as Backend>::Device::default();
        let base = std::path::PathBuf::from(path);
        use burn::record::NamedMpkFileRecorder;
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::default();
        type ModelRecord = <NeuralNetworkInner<AutoDiffCpuBackend> as burn::module::Module<
            AutoDiffCpuBackend,
        >>::Record;
        type OptimRecord = <OptimizerAdaptor<
            Adam,
            NeuralNetworkInner<AutoDiffCpuBackend>,
            AutoDiffCpuBackend,
        > as Optimizer<NeuralNetworkInner<AutoDiffCpuBackend>, AutoDiffCpuBackend>>::Record;
        let (model_rec, optim_rec, meta): (ModelRecord, OptimRecord, ModelMeta) =
            Recorder::load(&recorder, base, &device).map_err(|e| format!("load error: {e}"))?;
        let inner = NeuralNetworkInner::<AutoDiffCpuBackend>::new(&meta.sizes, &device)
            .load_record(model_rec);
        let optim = burn::optim::AdamConfig::new().init().load_record(optim_rec);
        Ok(NeuralNetwork::Cpu(NeuralNetworkCpu {
            device,
            inner,
            lr: meta.lr,
            optim,
            max_batch_size: meta.max_batch_size,
            best_loss: meta.best_loss,
            arch_sizes: meta.sizes,
        }))
    }

    pub fn load_with_backend(path: &str, backend: GpuBackend) -> Result<Self, String> {
        if try_init_gpu_device(backend).is_some() {
            let g = NeuralNetworkGpu::load_with_backend(path, backend)?;
            Ok(NeuralNetwork::Gpu(g))
        } else {
            // CPU path
            let device = <AutoDiffCpuBackend as Backend>::Device::default();
            let base = std::path::PathBuf::from(path);
            use burn::record::NamedMpkFileRecorder;
            let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::default();
            type ModelRecord = <NeuralNetworkInner<AutoDiffCpuBackend> as burn::module::Module<
                AutoDiffCpuBackend,
            >>::Record;
            type OptimRecord = <OptimizerAdaptor<
                Adam,
                NeuralNetworkInner<AutoDiffCpuBackend>,
                AutoDiffCpuBackend,
            > as Optimizer<
                NeuralNetworkInner<AutoDiffCpuBackend>,
                AutoDiffCpuBackend,
            >>::Record;
            let (model_rec, optim_rec, meta): (ModelRecord, OptimRecord, ModelMeta) =
                Recorder::load(&recorder, base, &device).map_err(|e| format!("load error: {e}"))?;
            let inner = NeuralNetworkInner::<AutoDiffCpuBackend>::new(&meta.sizes, &device)
                .load_record(model_rec);
            let optim = burn::optim::AdamConfig::new().init().load_record(optim_rec);
            Ok(NeuralNetwork::Cpu(NeuralNetworkCpu {
                device,
                inner,
                lr: meta.lr,
                optim,
                max_batch_size: meta.max_batch_size,
                best_loss: meta.best_loss,
                arch_sizes: meta.sizes,
            }))
        }
    }

    pub fn load_from_bytes(bytes: &[u8]) -> Result<Self, String> {
        Self::load_from_bytes_with_backend(bytes, GpuBackend::Default)
    }

    pub fn load_from_bytes_with_backend(bytes: &[u8], backend: GpuBackend) -> Result<Self, String> {
        if try_init_gpu_device(backend).is_some() {
            let g = NeuralNetworkGpu::load_from_bytes_with_backend(bytes, backend)?;
            Ok(NeuralNetwork::Gpu(g))
        } else {
            let c = NeuralNetworkCpu::load_from_bytes_with_backend(bytes)?;
            Ok(NeuralNetwork::Cpu(c))
        }
    }

    pub fn best_loss(&self) -> Option<f32> {
        match self {
            NeuralNetwork::Gpu(g) => g.best_loss,
            NeuralNetwork::Cpu(c) => c.best_loss,
        }
    }

    pub fn try_save_checkpoint(&mut self, path: &str, current_loss: f32) -> Result<bool, String> {
        let _is_better = self.best_loss().map(|b| current_loss < b).unwrap_or(true);
        // Delegate to inner variant and update its best_loss if improved
        match self {
            NeuralNetwork::Gpu(g) => {
                let is_better = g.best_loss.map(|b| current_loss < b).unwrap_or(true);
                if is_better {
                    g.best_loss = Some(current_loss);
                    g.save(path)?;
                    Ok(true)
                } else {
                    Ok(false)
                }
            }
            NeuralNetwork::Cpu(c) => {
                let is_better = c.best_loss.map(|b| current_loss < b).unwrap_or(true);
                if is_better {
                    c.best_loss = Some(current_loss);
                    c.save(path)?;
                    Ok(true)
                } else {
                    Ok(false)
                }
            }
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

    pub fn predict(&self, inputs: &[Vec<f32>], device: &B::Device) -> Vec<Vec<f32>> {
        // inputs: batch x (in_channels * 28 * 28)
        let batch = inputs.len();
        if batch == 0 {
            return Vec::new();
        }
        let flat_inputs: Vec<f32> = inputs.iter().flat_map(|v| v.iter().copied()).collect();
        let x = Tensor::<B, 4>::from_data(
            TensorData::new(flat_inputs, [batch, self.in_channels, 28, 28]),
            device,
        );
        let logits = self.forward_inner(x);
        let out_features = self.fc2.weight.shape().dims::<2>()[1];
        let vec = logits.into_data().to_vec().unwrap_or_default();
        let mut out: Vec<Vec<f32>> = Vec::with_capacity(batch);
        for chunk in vec.chunks(out_features) {
            out.push(chunk.to_vec());
        }
        out
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoke_train_and_predict_cpu_fallback_or_gpu() {
        let input_len = 32usize;
        let out_dim = 4usize;
        let num = 4usize;

        let mut inputs: Vec<Vec<f32>> = Vec::with_capacity(num);
        let mut targets: Vec<Vec<f32>> = Vec::with_capacity(num);
        for i in 0..num {
            inputs.push(vec![i as f32; input_len]);
            let mut oh = vec![0f32; out_dim];
            oh[i % out_dim] = 1.0;
            targets.push(oh);
        }

        let mut nn = NeuralNetwork::new(&[input_len, 8, out_dim], 2, 1e-3, GpuBackend::Default);
        let _loss = nn.train_batch(&inputs, &targets);
        let preds = nn.predict(&inputs);
        assert_eq!(preds.len(), num);
        assert_eq!(preds[0].len(), out_dim);
    }
}
