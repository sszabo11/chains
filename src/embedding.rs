use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, Mutex, MutexGuard},
};

use ndarray::{Array, Array1, Array2, Array3, ArrayBase, Dim, OwnedRepr};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use rand::{random, random_range};
use rayon::iter::{
    IntoParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelBridge,
    ParallelIterator,
};

use crate::read_harry_potters;

pub fn get_vocab(corpus: &[String]) -> HashMap<String, usize> {
    let mut v: HashMap<String, usize> = HashMap::new();

    for (i, line) in corpus.iter().enumerate() {
        let words: Vec<&str> = line.split_whitespace().collect();

        for (j, word) in words.iter().enumerate() {
            let word = tokenize_word(word);
            //let word: String = word
            //    .chars()
            //    .filter(|c| !c.is_ascii_punctuation())
            //    .map(|c| c.to_ascii_lowercase())
            //    .collect();

            if let Some(occurances) = v.get(&word) {
                v.insert(word, *occurances + 1);
            } else {
                v.insert(word, 1);
            }
        }
    }

    v
}

fn filter_infrequent_words(vocab: &HashMap<String, usize>) -> HashMap<String, usize> {
    let mut new_v = HashMap::new();
    for word in vocab.iter() {
        if *word.1 > 5 {
            new_v.insert(word.0.to_string(), *word.1);
        }
    }

    new_v
}

fn create_vocab_maps(
    v: &HashMap<String, usize>,
) -> (HashMap<String, usize>, HashMap<usize, String>) {
    let mut w_to_i: HashMap<String, usize> = HashMap::new();
    let mut i_to_w: HashMap<usize, String> = HashMap::new();

    for (i, word) in v.keys().enumerate() {
        w_to_i.insert(word.to_string(), i);
        i_to_w.insert(i, word.to_string());
    }

    (w_to_i, i_to_w)
}

pub struct Model {
    pub output_e: Arc<Mutex<ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>, f32>>>,
    pub input_e: Arc<Mutex<ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>, f32>>>,
    corpus: Vec<String>,
    pub vocab: HashMap<String, usize>, // With occurances
    w_to_i: HashMap<String, usize>,    // Word to index
    i_to_w: HashMap<usize, String>,    // Index to word
    dim: usize,
    sliding_window: usize,
    k: usize,
}

impl Model {
    pub fn new(corpus: &[String], dim: usize, sliding_window: usize, k: usize) -> Self {
        let vocab = get_vocab(corpus);
        let filtered_vocab = filter_infrequent_words(&vocab);
        let (w_to_i, i_to_w) = create_vocab_maps(&filtered_vocab);

        let vocab_size = vocab.len();
        Self {
            input_e: Arc::new(Mutex::new(Array::random(
                (vocab_size, dim),
                Uniform::new(-0.5, 0.5).unwrap(),
            ))),
            output_e: Arc::new(Mutex::new(Array::random(
                (vocab_size, dim),
                Uniform::new(-0.5, 0.5).unwrap(),
            ))),
            corpus: corpus.to_vec(),
            dim,
            vocab: filtered_vocab,
            w_to_i,
            k,
            sliding_window,
            i_to_w,
        }
    }

    pub fn embed(&self) {
        let vocab_size = self.vocab.len();

        //let mut a = Array::random((vocab_size, self.dim), Uniform::new(0.0, 1.0).unwrap());

        println!("Vocab: {:?}", self.w_to_i);

        println!("Vocab size: {}", vocab_size);

        let sliding_window = 1;

        for line in self.corpus.iter() {
            let words: Vec<&str> = line.split_whitespace().collect();

            for i in 0..words.len() {
                if i > 1 {
                    let word = tokenize_word(words[i]);

                    for w in 1..=sliding_window {
                        let word1 = tokenize_word(words[i - w]);
                        let word2 = tokenize_word(words[i + w]);
                        // self.compute_dot_product(&word, &word1);
                        // self.compute_dot_product(&word, &word2);
                    }
                }
            }
        }
    }

    pub fn train(&mut self, epochs: usize, learning_rate: f32, window: usize) {
        let e: Vec<_> = (0..epochs).collect();

        let output_e_clone = Arc::clone(&self.output_e);
        let input_e_clone = Arc::clone(&self.input_e);
        e.iter().for_each(|epoch| {
            println!("Epoch: {}", epoch);
            for line in self.corpus.iter() {
                let words: Vec<&str> = line.split_whitespace().collect();
                //println!("words: {:?}", words);

                for i in 0..words.len() {
                    if i.is_multiple_of(1000) {
                        println!("Status from epoch {}: 1000 words", epoch)
                    }
                    //println!("i: {}", i);
                    if i > self.k {
                        let target_word = tokenize_word(words[i]);
                        let target_idx = match self.w_to_i.get(&target_word) {
                            Some(idx) => idx,
                            None => continue,
                        };

                        'word: for w in
                            -(self.sliding_window as isize)..=self.sliding_window as isize
                        {
                            //println!("i w {} {} | {}", i, w, i as isize + w);
                            if (i as isize + w) as usize >= words.len() || w == 0 {
                                continue 'word;
                            };
                            let mut input_e = input_e_clone.lock().unwrap();
                            let mut output_e = output_e_clone.lock().unwrap();

                            let context_word = tokenize_word(words[(i as isize + w) as usize]);
                            let context_idx = match self.w_to_i.get(&context_word) {
                                Some(idx) => idx,
                                None => continue 'word,
                            };

                            println!("'{}' and '{}'", target_word, context_word);

                            // Similarity
                            let pos_dot_score = self.dot_product(*target_idx, *context_idx);

                            // Clamps 0-1
                            let pos_score = sigmoid(pos_dot_score);

                            // Error (want to be lower)

                            let pos_error = pos_score - 1.0;

                            // Temp vector for target input updates
                            let mut target_update = Array1::<f32>::zeros(input_e.ncols());

                            // Positive update to target
                            // Learning rate * positive error * output vec of context word
                            target_update +=
                                &(learning_rate * pos_error * &output_e.row(*context_idx));
                            // We do k negative samples at random
                            for _ in 0..self.k {
                                let neg_context_idx = random_range(0..self.i_to_w.len());
                                let neg_dot_score = self.dot_product(*target_idx, neg_context_idx);

                                let neg_score = sigmoid(neg_dot_score);
                                let neg_error = neg_score;

                                // How much to nudge
                                let update_factor = learning_rate * neg_error;

                                target_update += &(update_factor * &output_e.row(neg_context_idx));

                                let mut o_n = output_e.row_mut(neg_context_idx);

                                // Update negative output vector (push away)
                                o_n += &(update_factor * &input_e.row(*target_idx));
                            }

                            // Apply all accumalted updates to target input vector
                            let mut mut_target_input = input_e.row_mut(*target_idx);
                            mut_target_input += &target_update;

                            // Updates positive output vector (pull towards target)
                            let pos_update_factor = learning_rate * pos_error;

                            let mut mut_context_output = output_e.row_mut(*context_idx);
                            mut_context_output += &(pos_update_factor * &input_e.row(*target_idx));
                        }
                    }
                }
            }
        });
        println!("done");
    }

    fn dot_product(&self, target_idx: usize, context_idx: usize) -> f32 {
        let input_e = self.input_e.lock().unwrap();
        let output_e = self.output_e.lock().unwrap();
        input_e.row(target_idx).dot(&output_e.row(context_idx))
    }
}
fn tokenize_word(word: &str) -> String {
    let word: String = word
        .chars()
        .filter(|c| !c.is_ascii_punctuation())
        .map(|c| c.to_ascii_lowercase())
        .collect();
    word
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
