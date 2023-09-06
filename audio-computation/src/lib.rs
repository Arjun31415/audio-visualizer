use std::f32::consts::PI;

use fftw::{self, array::AlignedVec, plan::{R2CPlan, R2CPlan32}, types::{Flag, c32}};
#[allow(dead_code)]
pub struct Plan {
    fft_bassbuffer_size: i32,
    fft_midbuffer_size: i32,
    fft_treblebuffer_size: i32,
    number_of_bars: i32,
    audio_channels: u32,
    input_buffer_size: i32,
    rate: u32,
    bass_cut_off_bar: i32,
    treble_cut_off_bar: i32,
    sens_init: i32,
    autosens: i32,
    frame_skip: i32,
    status: i32,
    error_message: String,
    sens: f32,
    framerate: f32,
    noise_reduction: f32,

    p_bass_l: Option<R2CPlan32>,
    p_bass_r: Option<R2CPlan32>,
    p_mid_l: Option<R2CPlan32>,
    p_mid_r: Option<R2CPlan32>,
    p_treble_l: Option<R2CPlan32>,
    p_treble_r: Option<R2CPlan32>,

    out_bass_l: AlignedVec<c32>,
    out_bass_r: AlignedVec<c32>,
    out_mid_l: AlignedVec<c32>,
    out_mid_r: AlignedVec<c32>,
    out_treble_l: AlignedVec<c32>,
    out_treble_r: AlignedVec<c32>,

    bass_multiplier: Vec<f32>,
    mid_multiplier: Vec<f32>,
    treble_multiplier: Vec<f32>,

    in_bass_r_raw: AlignedVec<f32>,
    in_bass_l_raw: AlignedVec<f32>,
    in_mid_r_raw: AlignedVec<f32>,
    in_mid_l_raw: AlignedVec<f32>,
    in_treble_r_raw: AlignedVec<f32>,
    in_treble_l_raw: AlignedVec<f32>,
    in_bass_r: AlignedVec<f32>,
    in_bass_l: AlignedVec<f32>,
    in_mid_r: AlignedVec<f32>,
    in_mid_l: AlignedVec<f32>,
    in_treble_r: AlignedVec<f32>,
    in_treble_l: AlignedVec<f32>,

    prev_cava_out: Vec<f32>,
    cava_mem: Vec<f32>,
    input_buffer: Vec<f32>,
    cava_peak: Vec<f32>,

    eq: Vec<f32>,

    cut_off_frequency: Vec<f32>,
    FFTbuffer_upper_cut_off: Vec<i32>,
    FFTbuffer_lower_cut_off: Vec<i32>,
    cava_fall: Vec<f32>,
}
impl Default for Plan {
    fn default() -> Self {
        return Plan {
            ..Default::default()
        };
    }
}
impl Plan {
    fn new() -> Plan {
        return Plan {
            status: -1,
            ..Default::default()
        };
    }
}

fn init(
    number_of_bars: i32,
    rate: u32,
    channels: u32,
    autosens: i32,
    noise_reduction: f32,
    low_cut_off: i32,
    high_cut_off: i32,
) -> Plan {
    let mut p = Plan::new();
    p.status = 0;
    if channels < 1 || channels > 2 {
        p.error_message = format!("init called with illegal number of channels {}", channels);
        p.status = -1;
        return p;
    }
    if rate < 1 || rate > 384000 {
        p.error_message = format!("init called with illegal sample rate {}", rate);
        p.status = -1;
        return p;
    }
    let mut treble_buffer_size: u32 = 128;
    if rate > 8125 && rate <= 16250 {
        treble_buffer_size *= 2;
    } else if rate > 16250 && rate <= 32500 {
        treble_buffer_size *= 4;
    } else if rate > 32500 && rate <= 75000 {
        treble_buffer_size *= 8;
    } else if rate > 75000 && rate <= 150_000 {
        treble_buffer_size *= 16;
    } else if rate > 300_000 {
        treble_buffer_size *= 64;
    }
    if number_of_bars < 1 {
        p.error_message = format!(
            "init called with illegal number of bars {}. It must be a positive integer",
            number_of_bars
        );
        p.status = -1;
        return p;
    }
    if number_of_bars as u32 > treble_buffer_size / 2 + 1 {
        p.error_message = format!(
            "cava_init called with illegal number of bars: {}. For {} sample rate number of bars can't be more than {}",
            number_of_bars,
            rate,
            treble_buffer_size / 2 + 1
        );
        p.status = -1;
        return p;
    }
    if low_cut_off < 1 || high_cut_off < 1 {
        p.error_message = format!("low_cut_off must be a positive value");
        p.status = -1;
        return p;
    }
    if low_cut_off >= high_cut_off {
        p.error_message = format!("high_cut_off must be a higher than low_cut_off");
        p.status = -1;
        return p;
    }
    if high_cut_off as u32 > rate / 2 {
        p.error_message = format!(
            "high_cut_off can't be higher than sample rate / 2. (Nyquist Sampling Theorem)"
        );
        p.status = -1;
        return p;
    }
    p.number_of_bars = number_of_bars;
    p.audio_channels = channels;
    p.rate = rate;
    p.autosens = 1;
    p.sens_init = 1;
    p.sens = 1.0;
    p.autosens = autosens;
    p.framerate = 75.0;
    p.frame_skip = 1;
    p.noise_reduction = noise_reduction;

    p.fft_bassbuffer_size = treble_buffer_size as i32 * 8;
    p.fft_midbuffer_size = treble_buffer_size as i32 * 4;
    p.fft_treblebuffer_size = treble_buffer_size as i32;
    p.input_buffer_size = p.fft_bassbuffer_size * channels as i32;
    p.input_buffer = vec![0.0; p.input_buffer_size as usize];

    p.FFTbuffer_upper_cut_off = vec![0; number_of_bars as usize + 1];
    p.FFTbuffer_lower_cut_off = vec![0; number_of_bars as usize + 1];

    p.eq = vec![0.0; (number_of_bars as u32 * channels) as usize];
    p.cut_off_frequency = vec![0.0; (number_of_bars as u32 * channels) as usize];

    p.cava_fall = vec![0.0; (number_of_bars as u32 * channels) as usize];
    p.cava_mem = vec![0.0; (number_of_bars as u32 * channels) as usize];
    p.cava_peak = vec![0.0; (number_of_bars as u32 * channels) as usize];
    p.prev_cava_out = vec![0.0; (number_of_bars as u32 * channels) as usize];

    // Hann Window Calculation multipliers

    p.bass_multiplier = vec![0.0; p.fft_bassbuffer_size as usize];
    p.mid_multiplier = vec![0.0; p.fft_midbuffer_size as usize];
    p.treble_multiplier = vec![0.0; p.fft_treblebuffer_size as usize];
    for i in 0..p.fft_bassbuffer_size {
        p.bass_multiplier[i as usize] =
            0.5 * (1.0 - (2.0 * PI * (i / (p.fft_bassbuffer_size - 1)) as f32).cos());
    }
    for i in 0..p.fft_midbuffer_size {
        p.mid_multiplier[i as usize] =
            0.5 * (1.0 - (2.0 * PI * (i / (p.fft_midbuffer_size - 1)) as f32).cos());
    }
    for i in 0..p.fft_treblebuffer_size {
        p.treble_multiplier[i as usize] =
            0.5 * (1.0 - (2.0 * PI * (i / (p.fft_treblebuffer_size - 1)) as f32).cos());
    }
    // BASS
    let tmp_size = p.fft_bassbuffer_size as usize;
    p.in_bass_l = AlignedVec::new(tmp_size);
    p.in_bass_l_raw = AlignedVec::new(tmp_size);
    p.out_bass_l = AlignedVec::<c32>::new(tmp_size / 2 + 1);
    p.p_bass_l = Some(
        R2CPlan32::new(
            &[tmp_size],
            &mut p.in_bass_l,
            &mut p.out_bass_l,
            Flag::MEASURE,
        )
        .unwrap(),
    );

    // MID
    let tmp_size = p.fft_midbuffer_size as usize;
    p.in_mid_l = AlignedVec::new(tmp_size);
    p.in_bass_l_raw = AlignedVec::new(tmp_size);
    p.out_mid_l = AlignedVec::<c32>::new(tmp_size / 2 + 1);
    p.p_mid_l = Some(
        R2CPlan32::new(
            &[tmp_size],
            &mut p.in_mid_l,
            &mut p.out_mid_l,
            Flag::MEASURE,
        )
        .unwrap(),
    );

    // TREBLE
    let tmp_size = p.fft_treblebuffer_size as usize;
    p.in_treble_l = AlignedVec::new(tmp_size);
    p.in_treble_l_raw = AlignedVec::new(tmp_size);
    p.out_treble_l = AlignedVec::new(tmp_size / 2 + 1);
    p.p_treble_l = Some(
        R2CPlan32::new(
            &[tmp_size],
            &mut p.in_treble_l,
            &mut p.out_treble_l,
            Flag::MEASURE,
        )
        .unwrap(),
    );
    if p.audio_channels == 2 {
        // BASS
        let tmp_size = p.fft_bassbuffer_size as usize;
        p.in_bass_r = AlignedVec::new(tmp_size);
        p.in_bass_r_raw = AlignedVec::new(tmp_size);
        p.out_bass_r = AlignedVec::new(tmp_size / 2 + 1);
        p.p_bass_r = Some(
            R2CPlan32::new(
                &[tmp_size],
                &mut p.in_bass_r,
                &mut p.out_bass_r,
                Flag::MEASURE,
            )
            .unwrap(),
        );
        // MID
        let tmp_size = p.fft_midbuffer_size as usize;
        p.in_mid_r = AlignedVec::new(tmp_size);
        p.in_mid_r_raw = AlignedVec::new(tmp_size);
        p.out_mid_r = AlignedVec::new(tmp_size / 2 + 1);
        p.p_mid_r = Some(
            R2CPlan32::new(
                &[tmp_size],
                &mut p.in_mid_r,
                &mut p.out_mid_r,
                Flag::MEASURE,
            )
            .unwrap(),
        );
        // TREBLE
        let tmp_size = p.fft_treblebuffer_size as usize;
        p.in_treble_r = AlignedVec::new(tmp_size);
        p.in_treble_r_raw = AlignedVec::new(tmp_size);
        p.out_treble_r = AlignedVec::new(tmp_size / 2 + 1);
        p.p_treble_r = Some(
            R2CPlan32::new(
                &[tmp_size],
                &mut p.in_treble_r,
                &mut p.out_treble_r,
                Flag::MEASURE,
            )
            .unwrap(),
        );
    }
    p.input_buffer = vec![0.0; p.input_buffer_size as usize];
    p.cava_fall = vec![0.0; (number_of_bars as u32 * channels) as usize];
    p.cava_mem = vec![0.0; (number_of_bars as u32 * channels) as usize];
    p.prev_cava_out = vec![0.0; (number_of_bars as u32 * channels) as usize];
    // process: calculate cutoff frequencies and eq
    let lower_cut_off = low_cut_off;
    let upper_cut_off = high_cut_off;
    let bass_cut_off = 100;
    let treble_cut_off = 500;
    let frequency_constant: f32 = (lower_cut_off as f32 / upper_cut_off as f32).log10()
        / (1.0 / (p.number_of_bars as f32 + 1.0) - 1.0);
    let mut relative_cut_off = vec![0; p.number_of_bars as usize + 1];
    p.bass_cut_off_bar = -1;
    p.treble_cut_off_bar = -1;
    let first_bar = 1;
    let first_treble_bar = 0;
    let bar_buffer = vec![0; p.number_of_bars as usize + 1];
    for n in 0..p.number_of_bars as usize + 1 {
        let bar_distribution_coeff = -frequency_constant
            + ((n + 1) as f32) / ((p.number_of_bars + 1) as f32) * frequency_constant;
        p.cut_off_frequency[n] = upper_cut_off as f32 * 10.0_f32.powf(bar_distribution_coeff);
        if n > 0 {
            if p.cut_off_frequency[n - 1] >= p.cut_off_frequency[n]
                && p.cut_off_frequency[n - 1] > bass_cut_off as f32
            {
                p.cut_off_frequency[n] = p.cut_off_frequency[n - 1]
                    + (p.cut_off_frequency[n - 1] - p.cut_off_frequency[n - 2]);
            }
        }
        relative_cut_off[n] = (p.cut_off_frequency[n] / (p.rate as f32 / 2.0)) as i32;
        p.eq[n] = p.cut_off_frequency[n];
        p.eq[n] /= 2.0_f32.powi(29);
        p.eq[n] /= p.fft_bassbuffer_size.ilog2() as f32;

    }

    return p;
}
fn execute(in_buffer: Vec<f32>, new_samples: i32, out_buffer: Vec<f32>, plan: Plan) {}
fn destroy(plan: Plan) {}
