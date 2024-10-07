use burn::{config::Config, data::{dataloader::DataLoaderBuilder, dataset::InMemDataset}, module::Module, optim::AdamWConfig, record::CompactRecorder, tensor::backend::AutodiffBackend, train::{metric::{AccuracyMetric, LossMetric}, LearnerBuilder}};
use crate::bert::{BertForPreTrainingConfig, BertPreTrainingBatcher, BertPreTrainingBatcherBuilder, BertPreTrainingInputItem};

#[derive(Config)]
pub struct TrainingConfig {
    pub model: BertForPreTrainingConfig,
    pub optimizer: AdamWConfig,
    pub artifact_dir: String, 
    #[config(default =  0)]
    pub mask_token_id: usize,
    #[config(default = 10)]
    pub num_epochs: usize,
    #[config(default = 2)]
    pub batch_size: usize,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1.0e-5)]
    pub learning_rate: f64,
}

fn create_artifact_dir(artifact_dir: &str) {
    // Remove existing artifacts before to get an accurate learner summary
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

pub fn train<B: AutodiffBackend>(
    config: TrainingConfig, 
    train: InMemDataset<BertPreTrainingInputItem>,
    valid: InMemDataset<BertPreTrainingInputItem>,
    device: B::Device) 
{
    let artifact_dir = &config.artifact_dir;
    create_artifact_dir(artifact_dir);

    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Config should be saved successfully");

    B::seed(config.seed);

    let batcher_train: BertPreTrainingBatcher<B> = BertPreTrainingBatcherBuilder::default()
        .mask_token_id(config.mask_token_id as u32)
        .pad_token_id(config.model.pad_token_id as u32)
        .voc_size(config.model.vocab_size)
        .max_seq_length(config.model.max_position_embeddings as u32)
        .device(device.clone())
        .build()
        .unwrap();

    let batcher_valid: BertPreTrainingBatcher<B::InnerBackend> = BertPreTrainingBatcherBuilder::default()
        .mask_token_id(config.mask_token_id as u32)
        .pad_token_id(config.model.pad_token_id as u32)
        .voc_size(config.model.vocab_size)
        .max_seq_length(config.model.max_position_embeddings as u32)
        .device(device.clone())
        .build()
        .unwrap();

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(train);

    let dataloader_test = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(valid);

    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .summary()
        .build(
            config.model.init::<B>(&device),
            config.optimizer.init(),
            config.learning_rate,
        );

    let model_trained = learner.fit(dataloader_train, dataloader_test);

    model_trained
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Trained model should be saved successfully");
}
