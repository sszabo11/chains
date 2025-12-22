use std::{fs, io::Write, time::UNIX_EPOCH};

use bhtsne::tSNE;
use ndarray::{ArrayBase, Dim, Ix2, OwnedRepr};
use plotters::prelude::*;

pub fn draw_graph(input: Vec<f32>, dim: u8, epochs: usize) {
    let flattened = flatten(input, dim, epochs);
}

const PERPLEXITY: f32 = 20.0;
const THETA: f32 = 0.5;

fn flatten(input: Vec<f32>, dim: u8, epochs: usize) -> Vec<f32> {
    tSNE::new(&input)
        .embedding_dim(dim)
        .perplexity(PERPLEXITY)
        .epochs(epochs)
        .barnes_hut(THETA, |a, b| (a - b).powi(2).sqrt())
        .embedding()
}

//pub fn draw(
//    words: Vec<String>,
//    data: ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>, f32>,
//    dim: u8,
//    epochs: usize,
//) -> Result<(), Box<dyn std::error::Error>> {
//    let root = BitMapBackend::new("./graph/0.png", (640, 480)).into_drawing_area();
//    root.fill(&WHITE)?;
//
//    let mut chart = ChartBuilder::on(&root)
//        .caption("Embedding visualized in 2D", ("sans-serif", 30).into_font())
//        .margin(5)
//        .x_label_area_size(30)
//        .y_label_area_size(30)
//        .build_cartesian_2d((0u32..10u32).into_segmented(), 0u32..10u32)?;
//
//    chart
//        .configure_mesh()
//        .disable_x_mesh()
//        .disable_y_mesh()
//        .bold_line_style(WHITE.mix(0.3))
//        .axis_desc_style(("sans-serif", 15))
//        .draw()?;
//
//    //chart.draw_series(
//    //    Histogram::vertical(&chart)
//    //        .style(RED.mix(0.5).filled())
//    //        .data(data.iter()),
//    //)?;
//
//    let v = data.into_raw_vec_and_offset().0;
//    let tsne: Vec<(f32, f32)> = tSNE::new(&v)
//        .embedding_dim(dim)
//        .perplexity(PERPLEXITY)
//        .epochs(epochs)
//        .barnes_hut(THETA, |a, b| (*a, *b))
//        .embedding();
//
//    for (i, word) in words.iter().enumerate() {
//        let x = data[[i, 0]] as u32;
//        let y = data[[i, 1]] as u32;
//
//        let circle = Circle::new((x, y), 2, ShapeStyle::from(RED).filled());
//
//        let label = Text::new(word.clone(), (x, y), ("sans-serif", 12).into_font());
//
//        chart.draw_series(std::iter::once(&circle)).unwrap();
//        chart.draw_series(std::iter::once(&label)).unwrap();
//    }
//
//    //chart
//    //    .configure_series_labels()
//    //    .background_style(WHITE.mix(0.8))
//    //    .border_style(BLACK)
//    //    .draw()?;
//
//    root.present()?;
//
//    Ok(())
//}

pub fn draw2(
    words: &[String],
    data: ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
    img_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let time = std::time::SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let path = format!("./graph/{}.png", time);
    println!(
        "file:///home/rabbit/Desktop/rusty/markov{}",
        path.split_once(".").unwrap().1
    );
    //let _ = fs::File::create_new(&path).unwrap();

    let img = format!("./graph/{}", img_path);
    let root = BitMapBackend::new(&img, (1200, 1200)).into_drawing_area();
    root.fill(&WHITE)?;

    // Perform t-SNE reduction
    let samples: Vec<Vec<f32>> = data.rows().into_iter().map(|row| row.to_vec()).collect();

    let points: Vec<f32> = tSNE::new(&samples)
        .embedding_dim(2)
        .perplexity(PERPLEXITY)
        .learning_rate(200.0)
        .epochs(2000)
        .barnes_hut(THETA, |a, b| {
            a.iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).powi(2))
                .sum::<f32>()
                .sqrt()
        })
        .embedding();

    let xs: Vec<f32> = points
        .iter()
        .copied()
        .enumerate()
        .filter(|(i, _)| i % 2 == 0)
        .map(|(_, x)| x)
        .collect();

    let ys: Vec<f32> = points
        .iter()
        .copied()
        .enumerate()
        .filter(|(i, _)| i % 2 == 1)
        .map(|(_, y)| y)
        .collect();

    let padding = 10.0;

    let x_min = xs.iter().cloned().fold(f32::INFINITY, f32::min) - padding;

    let x_max = xs.iter().cloned().fold(f32::NEG_INFINITY, f32::max) + padding;

    let y_min = ys.iter().cloned().fold(f32::INFINITY, f32::min) - padding;

    let y_max = ys.iter().cloned().fold(f32::NEG_INFINITY, f32::max) + padding;

    let mut chart = ChartBuilder::on(&root)
        .caption("Embedding visualized in 2D", ("sans-serif", 30).into_font())
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(x_min..x_max, y_min..y_max)?;

    chart
        .configure_mesh()
        .disable_x_mesh()
        .disable_y_mesh()
        .bold_line_style(WHITE.mix(0.3))
        .axis_desc_style(("sans-serif", 15))
        .draw()?;

    for ((x, y), word) in xs.iter().zip(ys.iter()).zip(words.iter()) {
        //let x = (*x * SCALER).round() as u32;
        //let y = (*y * SCALER).round() as u32;

        let circle = Circle::new((*x, *y), 2, ShapeStyle::from(RED).filled());
        let label = Text::new(word.clone(), (*x, *y), ("sans-serif", 12).into_font());
        chart.draw_series(std::iter::once(circle))?;
        chart.draw_series(std::iter::once(label))?;
    }

    root.present()?;
    Ok(())
}

const SCALER: f32 = 10.0;
