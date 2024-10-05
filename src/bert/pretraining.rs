use burn::{data::dataloader::batcher::Batcher, nn::attention::{generate_padding_mask, GeneratePaddingMask}, prelude::Backend, tensor::{Distribution, Int, Shape, Tensor, TensorData}};
use derive_builder::Builder;
use tokenizers::Encoding;

use super::BertInputBatch;

/// One single input item for the bert pretraining. 
/// It comprises both a tokenized sentence pair and one nsp label
#[derive(Debug, Clone, Builder)]
pub struct BertPreTrainingInputItem {
    /// the tokenized sentence pair
    pub encoding: Encoding,
    /// the next sequence prediction label
    pub nsp_label: u32,
}

/// The input of a bert model.
/// 
/// The only mandatory input is the 'input_ids' field.
#[derive(Debug, Clone, Builder)]
pub struct BertPreTrainingLabelBatch<B: Backend> {
    /// The labels for the MLM task
    /// Shape: [batch, sequence]
    pub mlm_labels: Tensor<B, 2, Int>,
    /// The labels for the NSP task 
    /// Shape: [batch, sequence]
    pub nsp_labels: Tensor<B, 1, Int>,
}

/// The input for a bert pretraining step.
/// 
/// The only mandatory input is the 'input_ids' field.
#[derive(Debug, Clone, Builder)]
pub struct BertPreTrainingInputBatch<B: Backend> {
    pub input: BertInputBatch<B>,
    pub labels: BertPreTrainingLabelBatch<B>,
}

/// Creates a BertInputBatch from a series of bert pretraining inputs 
/// (that is, from a series of tokenized sequence pairs alongside with an nsp label)
#[derive(Debug, Clone, Builder)]
pub struct BertPreTrainingBatcher<B: Backend> {
    /// probability that a token be replaced by this processor
    #[builder(default="0.15")]
    pub mlm_proba: f64,
    /// the identifier of the [MASK] token in the vocabulary
    /// => for bert-base-uncased, it would be 103
    pub mask_token_id: u32,
    /// the identifier of the [PAD] token in the vocabulary
    /// => for bert-base-uncased, it would be 0
    pub pad_token_id: u32,
    /// maximum sequence length for the tokenized text
    pub max_seq_length: u32,
    /// the size of the vocabulary
    /// => for bert-base-uncased, it would be 30522
    pub voc_size: usize,
    /// the device where the tensors must be created
    pub device: B::Device
}

impl <B: Backend> BertPreTrainingBatcher<B> {
    /// Automatically applies MLM for the BERT pretraining.
    pub fn mask_tokens(&self, mut input_ids: Tensor<B, 2, Int>, special_tokens: Tensor<B, 2, Int>) -> (Tensor<B, 2, Int>, Tensor<B, 2, Int>){
        let shape = input_ids.shape();
        let device = input_ids.device();

        // this creates a boolean tensor of the same shape (but so far, it is considered to be a float tensor)
        let replaced: Tensor<B, 2, Int> = Tensor::random(shape.clone(), Distribution::Bernoulli(self.mlm_proba), &device);
        let replaced = replaced.mask_fill(special_tokens.clone().bool(), 0.0);
        let labels = input_ids.clone().mask_fill(replaced.clone().bool().bool_not(), -100);

        // 80% of the replaced tokens will be replaced by [MASK]
        let masked: Tensor<B, 2, Int> = Tensor::random(shape.clone(), Distribution::Bernoulli(0.8), &device);
        let masked = masked.min_pair(replaced.clone());
        input_ids = input_ids.mask_fill(masked.clone().bool(), self.mask_token_id);

        // 10% of the replaced tokens will be replaced by some random token
        let randtoken: Tensor<B, 2, Int> = Tensor::random(shape.clone(), Distribution::Bernoulli(0.5), &device);
        let randtoken = randtoken.min_pair(replaced.clone());
        let randtoken = randtoken.min_pair(masked.clone().bool().bool_not().int());
        let rnd_words: Tensor<B, 2, Int> = Tensor::random(shape.clone(), Distribution::Uniform(0.0, self.voc_size as f64), &device);
        input_ids = input_ids.mask_where(randtoken.clone().bool(), rnd_words);

        // 10% remaining masked tokens will actually be left unchanged
        (input_ids, labels)
    }

    /// Given an iterator over slices of integers, this method will create a 2d tensor of the given shape.
    /// Whenever a sequence from the iterator is shorter than shape[1], padding will be used.
    fn to_tensor_of_shape<'a, I>(&self, shape: Shape<2>, iter: I) -> Tensor<B, 2, Int>
        where I: Iterator<Item = &'a [u32]> + 'a
    {
        let mut tensor: Tensor<B, 2, Int> = Tensor::full(shape, self.pad_token_id, &self.device);
        
        for (i, item) in iter.enumerate() {
            tensor = tensor.slice_assign(
                #[allow(clippy::single_range_in_vec_init)]
                [i..i+1], 
                Tensor::from_data(
                    TensorData::new(item.to_vec(), [1, item.len()]), 
                    &self.device));
        }

        tensor
    }
}

impl <B: Backend> Batcher<BertPreTrainingInputItem, BertPreTrainingInputBatch<B>> for BertPreTrainingBatcher<B> {
    fn batch(&self, items: Vec<BertPreTrainingInputItem>) -> BertPreTrainingInputBatch<B> {
        let tokens_list = items.iter()
            .map(|i| i.encoding.get_ids().iter().copied().map(|i| i as usize).collect())
            .collect();

        // let us first ensure that all sequences are padded to a common length
        let GeneratePaddingMask{tensor, mask} = generate_padding_mask::<B>(
            self.pad_token_id as usize, 
            tokens_list, 
            Some(self.max_seq_length as usize), 
            &self.device);

        let input_ids = tensor;
        let padding_mask = mask;
        let shape = input_ids.shape();

        let token_type_ids = self.to_tensor_of_shape(shape.clone(), items.iter().map(|i| i.encoding.get_type_ids()));
        let special_tokens = self.to_tensor_of_shape(shape.clone(), items.iter().map(|i| i.encoding.get_special_tokens_mask()));

        let nsp_labels = items.iter().map(|i| i.nsp_label).collect::<Vec<_>>();
        let nsp_labels_len = nsp_labels.len();
        let nsp_labels = Tensor::from_data(TensorData::new(nsp_labels, [nsp_labels_len]), &self.device);

        let (input_ids, mlm_labels) = self.mask_tokens(input_ids, special_tokens);

        BertPreTrainingInputBatch {
            input: BertInputBatch {
                input_ids,
                token_type_ids: Some(token_type_ids),
                padding_mask: Some(padding_mask),
            },
            labels: BertPreTrainingLabelBatch { 
                mlm_labels,
                nsp_labels,
            },
        }
    }
}



#[cfg(test)]
pub mod test {
    use burn::{backend::{wgpu::WgpuDevice, Wgpu}, data::dataloader::batcher::Batcher};
    use tokenizers::Tokenizer;

    use crate::bert::{BertMLMHeadConfig, BertModelConfig};

    use super::{BertPreTrainingBatcherBuilder, BertPreTrainingInputItemBuilder};

    #[test]
    fn test_tokenizer() {
        let device = WgpuDevice::IntegratedGpu(0);
        let tok = Tokenizer::from_pretrained("xaviergillard/parti-pris-v2", None).unwrap();

        let mask_id = tok.get_added_vocabulary().get_vocab()["[MASK]"];
        let pad_id=  tok.get_added_vocabulary().get_vocab()["[PAD]"];
        
        let batcher = BertPreTrainingBatcherBuilder::<Wgpu>::default()
            .pad_token_id(pad_id)
            .mask_token_id(mask_id)
            .max_seq_length(15)
            .voc_size(tok.get_vocab_size(false))
            .device(device.clone())
            .build()
            .unwrap();

        let text1 = "Bonjour Xavier, comment vas-tu aujourd'hui ?";
        let encoded1 = tok.encode(text1, true).unwrap();
        let item1 = BertPreTrainingInputItemBuilder::default()
            .encoding(encoded1)
            .nsp_label(0)
            .build()
            .unwrap();

        let text2 = "Une phrase courte";
        let encoded2 = tok.encode(text2, true).unwrap();
        let item2 = BertPreTrainingInputItemBuilder::default()
            .encoding(encoded2)
            .nsp_label(0)
            .build()
            .unwrap();


        let batch = batcher.batch(vec![item1, item2]);

        let model = BertModelConfig::new().init::<Wgpu>(&device);
        let hidden = model.forward(batch.input);
        let head = BertMLMHeadConfig::new().init(&device);
        let hidden = head.forward(hidden);

        println!("{hidden:?}");

    }
}