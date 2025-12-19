use std::{
    collections::HashMap,
    fs::{self, File},
    io, process,
};
mod embedding;

use anyhow::Result;
use rand::Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::embedding::Model;

fn main() {
    //let corpus = fs::read_to_string("./corpus/harry-potter-1").unwrap();

    let harry = read_harry_potters();
    let seuss = parse_seuss();

    println!("Corpus len: {}", harry.len());
    //println!("Corpus words: {}", harry.join(" ").split_whitespace().collect::<Vec<&str>>().len());

    let mut model = Model::new(&seuss, 300, 4, 10);
    println!("{:?}", model.vocab);

    model.train(10, 0.025, 0);

    println!("Success!");
    println!("{}", model.input_e.lock().unwrap())

    //let yt = parse_yt();
    //let moby = read_txt_file("./corpus/moby.txt");
    //let seuss = parse_seuss();
    //let imdb = parse_imdb("./corpus/imdb.csv", 0).unwrap();
    //let jeopardy = parse_imdb("./corpus/jeopardy.csv", 7).unwrap();

    //let moby_lines: Vec<String> = moby
    //    .lines()
    //    .map(|s| s.trim().to_string())
    //    .filter(|s| !s.is_empty())
    //    .collect();

    //let mut all_texts: Vec<String> = Vec::new();
    //all_texts.extend(imdb);
    //all_texts.extend(jeopardy);
    //all_texts.extend(harry);
    //all_texts.extend(moby_lines);
    //all_texts.extend(seuss);
    //all_texts.extend(yt);

    //const DEGREE: usize = 2;
    //let mut chain = all_texts
    //    .par_iter()
    //    .map(|text| {
    //        let mut local_chain = Chain::new(DEGREE);
    //        local_chain.process(text);
    //        local_chain
    //    })
    //    .reduce(
    //        || Chain::new(DEGREE),
    //        |mut a, b| {
    //            a.merge(b);
    //            a
    //        },
    //    );

    //chain.process(&moby);
    ////chain.process(&yt);
    //println!("Text len: {}", all_texts.len());
    //println!("Chain len: {}", chain.map.len());

    ////println!("Values: {:?}", chain.map);

    ////let text = chain.predict(200);
    //let question = String::from("What is the best movie?");
    //let text = chain.question(question, 100);
    //println!("\n \n{}", text);
}

#[derive(Debug, Serialize, Deserialize)]
struct YoutubeRecord {
    #[serde(rename = "Comment")]
    comment: String,
    #[serde(rename = "Sentiment")]
    sentiment: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct IMDBRecord {
    review: String,
    sentiment: String,
}

struct Chain {
    map: HashMap<String, Vec<String>>,
    degree: usize,
}

fn read_txt_file(path: &str) -> String {
    fs::read_to_string(path).unwrap()
}

fn parse_yt() -> Vec<String> {
    let mut rdr = csv::ReaderBuilder::new()
        .buffer_capacity(70 * 1024 * 1024) // 70 mib
        .from_path("./corpus/youtube.csv")
        .unwrap();

    //let mut data: Vec<YoutubeRecord> = Vec::new();

    let mut texts = Vec::new();
    for res in rdr.deserialize() {
        let rec: YoutubeRecord = res.unwrap();
        texts.push(rec.comment);
    }

    texts
}

fn parse_imdb(path: &str, text_col: usize) -> Result<Vec<String>> {
    let file = File::open(path)?;

    let mut texts = Vec::new();
    let mut rdr = csv::ReaderBuilder::new().flexible(true).from_reader(file);

    for result in rdr.records() {
        let record = result?;
        if let Some(field) = record.get(text_col) {
            let cleaned = field.trim().to_string();
            if !cleaned.is_empty() {
                texts.push(cleaned);
            }
        }
    }
    //./target/debug/markov < corpus/imdb.csv  376.49s user 0.29s system 99% cpu 6:17.67 total

    Ok(texts)
}
fn parse_moby() -> String {
    fs::read_to_string("./corpus/moby.txt").unwrap()
}

fn parse_seuss() -> Vec<String> {
    let dir = fs::read_dir("./corpus/seuss").unwrap();
    let mut texts = Vec::new();
    for entry in dir {
        let path = entry.unwrap().path();

        if path.is_file() {
            let content = fs::read_to_string(&path).unwrap();
            texts.extend(content.lines().map(|l| l.trim().to_string()));
        }
    }

    texts
}

impl Chain {
    fn new(degree: usize) -> Self {
        Self {
            degree,
            map: HashMap::new(),
        }
    }

    fn merge(&mut self, other: Chain) {
        for (key, mut values) in other.map {
            self.map.entry(key).or_default().append(&mut values);
        }
    }

    fn question(&mut self, question: String, len: usize) -> String {
        self.process(&question);
        let mut rng = rand::rng();

        let question = self.remove_punc(question);
        let question_words: Vec<&str> = question.split_whitespace().collect();

        let mut last_words = String::new();
        let mut d = self.degree;
        while d > 0 {
            let word = question_words[question_words.len() - d];
            last_words.push_str(word);
            if d != 1 {
                last_words.push(' ');
            }
            d -= 1;
        }

        let mut text = String::new();

        for _ in 0..len {
            // "rake and" -> ["dig", ...]
            if let Some(options) = self.map.get(&last_words) {
                let o = rng.random_range(0..options.len());

                //println!("Last words: {}", last_words);
                // rake and -> ["rake", "and"]
                let curr_words: Vec<&str> = last_words.split(" ").collect();

                // "rake and"
                let next_word = options[o].clone();
                text.push(' ');
                text.push_str(&next_word);

                assert!(curr_words.len() == self.degree);

                let mut next = String::new();
                //println!("cur: {:?}", curr_words);
                for i in 1..self.degree {
                    let r = curr_words[curr_words.len() - i];
                    next.push_str(r);
                    next.push(' ');
                }
                next.push_str(&next_word);

                last_words = next;
            } else {
                eprintln!("Word not found in map: |{}|", last_words);
                break;
            };
        }

        text
    }

    fn remove_punc(&self, mut text: String) -> String {
        text.retain(|c| !c.is_ascii_punctuation());

        text
    }

    fn process(&mut self, text: &str) {
        let words: Vec<&str> = text.split_whitespace().collect();

        for i in self.degree..words.len() {
            let mut key = String::new();
            let mut d = self.degree;
            while d > 0 {
                let word = words[i - d];
                key.push_str(word);
                if d != 1 {
                    key.push(' ');
                }
                d -= 1;
            }

            //let first_word = words[i - 2];
            //let second_word = words[i - 1];
            let next_word = words[i];

            //let key = format!("{} {}", first_word, second_word);

            let prev = self.map.get_mut(&key);

            let value = if let Some(prev) = prev {
                prev.push(next_word.to_string());
                prev.to_vec()
            } else {
                vec![next_word.to_string()]
            };

            self.map.insert(key, value);
        }
    }

    fn predict(&self, len: usize) -> String {
        let mut rng = rand::rng();

        let i = rng.random_range(0..self.map.len());

        let entries: Vec<_> = self.map.keys().collect();

        let mut last_words: String = entries[i].clone();
        let mut text = String::new();

        println!("last words {}", last_words);
        for _ in 0..len {
            if let Some(options) = self.map.get(&last_words) {
                let o = rng.random_range(0..options.len());

                let curr_words: Vec<&str> = last_words.split(" ").collect();

                let next_word = options[o].clone();
                text.push(' ');
                text.push_str(&next_word);

                assert!(curr_words.len() == self.degree);

                let mut next = String::new();
                for i in 1..self.degree {
                    let r = curr_words[curr_words.len() - i];
                    next.push_str(r);
                    next.push(' ');
                }
                next.push_str(&next_word);

                last_words = next;
            } else {
                eprintln!("Word not found in map: |{}|", last_words);
                break;
            };
        }

        text
    }
}

fn read_harry_potters() -> Vec<String> {
    let mut lines = Vec::new();

    for i in 1..=7 {
        let path = format!("./corpus/harry-potter-{}", i);
        let content = fs::read_to_string(&path).unwrap_or_else(|e| {
            eprintln!("Failed to read {}: {}", path, e);
            String::new()
        });

        lines.extend(
            content
                .lines()
                .map(|line| line.trim().to_string())
                .filter(|line| !line.is_empty()),
        );
    }

    lines
}
