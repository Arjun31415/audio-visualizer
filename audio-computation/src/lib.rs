use std::{
    cmp::{max, min},
    f32::consts::PI,
};

use fftw::{
    self,
    array::AlignedVec,
    plan::{R2CPlan, R2CPlan32},
    types::{c32, Flag},
};
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
    sens_init: bool,
    autosens: bool,
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
            fft_bassbuffer_size: 0,
            fft_midbuffer_size: 0,
            fft_treblebuffer_size: 0,
            number_of_bars: 0,
            audio_channels: 0,
            input_buffer_size: 0,
            rate: 0,
            bass_cut_off_bar: 0,
            treble_cut_off_bar: 0,
            sens_init: false,
            autosens: false,
            frame_skip: 0,
            status: 0,
            error_message: "".to_string(),
            sens: 0_f32,
            framerate: 0_f32,
            noise_reduction: 0_f32,
            p_bass_l: None,
            p_bass_r: None,
            p_mid_l: None,
            p_mid_r: None,
            p_treble_l: None,
            p_treble_r: None,
            out_bass_l: AlignedVec::new(0),
            out_bass_r: AlignedVec::new(0),
            out_mid_l: AlignedVec::new(0),
            out_mid_r: AlignedVec::new(0),
            out_treble_l: AlignedVec::new(0),
            out_treble_r: AlignedVec::new(0),
            bass_multiplier: vec![0_f32; 0],
            mid_multiplier: vec![0_f32; 0],
            treble_multiplier: vec![0_f32; 0],
            in_bass_r_raw: AlignedVec::new(0),
            in_bass_l_raw: AlignedVec::new(0),
            in_mid_r_raw: AlignedVec::new(0),
            in_mid_l_raw: AlignedVec::new(0),
            in_treble_r_raw: AlignedVec::new(0),
            in_treble_l_raw: AlignedVec::new(0),
            in_bass_r: AlignedVec::new(0),
            in_bass_l: AlignedVec::new(0),
            in_mid_r: AlignedVec::new(0),
            in_mid_l: AlignedVec::new(0),
            in_treble_r: AlignedVec::new(0),
            in_treble_l: AlignedVec::new(0),
            prev_cava_out: vec![0_f32; 0],
            cava_mem: vec![0_f32; 0],
            input_buffer: vec![0_f32; 0],
            cava_peak: vec![0_f32; 0],

            eq: vec![0_f32; 0],
            cut_off_frequency: vec![0_f32; 0],
            FFTbuffer_upper_cut_off: vec![0; 0],
            FFTbuffer_lower_cut_off: vec![0; 0],
            cava_fall: vec![0_f32; 0],
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
    autosens: bool,
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
    p.autosens = true;
    p.sens_init = true;
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
        let temp = 0.5 * (1.0 - (2.0 * PI * i as f32 / (p.fft_bassbuffer_size - 1) as f32).cos());
        p.bass_multiplier[i as usize] = temp;
    }
    for i in 0..p.fft_midbuffer_size {
        p.mid_multiplier[i as usize] =
            0.5 * (1.0 - (2.0 * PI * i as f32 / (p.fft_midbuffer_size - 1) as f32).cos());
    }
    for i in 0..p.fft_treblebuffer_size {
        p.treble_multiplier[i as usize] =
            0.5 * (1.0 - (2.0 * PI * i as f32 / (p.fft_treblebuffer_size - 1) as f32).cos());
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
    p.in_mid_l_raw = AlignedVec::new(tmp_size);
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
    let mut relative_cut_off = vec![0.0; p.number_of_bars as usize + 1];
    p.bass_cut_off_bar = -1;
    p.treble_cut_off_bar = -1;
    let mut first_bar = 1;
    let mut first_treble_bar = 0;
    let mut bar_buffer = vec![0; p.number_of_bars as usize + 1];

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
        relative_cut_off[n] = p.cut_off_frequency[n] / (p.rate as f32 / 2.0);
        p.eq[n] = p.cut_off_frequency[n];
        p.eq[n] /= 2.0_f32.powi(29);
        p.eq[n] /= p.fft_bassbuffer_size.ilog2() as f32;
        if p.cut_off_frequency[n] < bass_cut_off as f32 {
            bar_buffer[n] = 1;
            p.FFTbuffer_lower_cut_off[n] =
                (relative_cut_off[n] * (p.fft_bassbuffer_size / 2) as f32) as i32;
            p.bass_cut_off_bar += 1;
            p.treble_cut_off_bar += 1;
            if p.bass_cut_off_bar > 0 {
                first_bar = 0;
            }
            p.FFTbuffer_lower_cut_off[n] =
                min(p.FFTbuffer_lower_cut_off[n], p.fft_bassbuffer_size / 2);
        } else if p.cut_off_frequency[n] > bass_cut_off as f32
            && p.cut_off_frequency[n] < treble_cut_off as f32
        {
            bar_buffer[n] = 2;
            p.FFTbuffer_lower_cut_off[n] =
                (relative_cut_off[n] * (p.fft_midbuffer_size as f32 / 2.0)) as i32;
            p.treble_cut_off_bar += 1;
            if p.treble_cut_off_bar - p.bass_cut_off_bar == 1 {
                first_bar = 1;
                if n > 0 {
                    p.FFTbuffer_upper_cut_off[n - 1] =
                        (relative_cut_off[n] * (p.fft_bassbuffer_size as f32 / 2.0)) as i32;
                }
            } else {
                first_bar = 0;
            }
            p.FFTbuffer_lower_cut_off[n] =
                min(p.FFTbuffer_lower_cut_off[n], p.fft_midbuffer_size / 2);
        } else {
            // TREBLE
            bar_buffer[n] = 3;
            p.FFTbuffer_lower_cut_off[n] =
                (relative_cut_off[n] * (p.fft_treblebuffer_size as f32 / 2.0)) as i32;
            first_treble_bar += 1;
            if first_treble_bar == 1 {
                first_bar = 1;
                if n > 0 {
                    p.FFTbuffer_upper_cut_off[n - 1] =
                        (relative_cut_off[n] * (p.fft_midbuffer_size as f32 / 2.0)) as i32;
                }
            } else {
                first_bar = 0;
            }
            p.FFTbuffer_lower_cut_off[n] =
                min(p.FFTbuffer_lower_cut_off[n], p.fft_treblebuffer_size / 2);
        }
        if n > 0 {
            // dbg!(first_bar);
            if first_bar == 0 {
                p.FFTbuffer_upper_cut_off[n - 1] = p.FFTbuffer_lower_cut_off[n] - 1;
                // pushing the spectrum up if the exponential function gets "clumped" in the
                // bass and caluclating new cut off frequencies
                /* dbg!(
                    &p.FFTbuffer_lower_cut_off[n],
                    &p.FFTbuffer_lower_cut_off[n - 1]
                ); */
                if p.FFTbuffer_lower_cut_off[n] <= p.FFTbuffer_lower_cut_off[n - 1] {
                    // dbg!(123);
                    let mut room_for_more: bool = false;
                    if bar_buffer[n] == 1 {
                        if p.FFTbuffer_lower_cut_off[n - 1] + 1 < p.fft_bassbuffer_size / 2 + 1 {
                            room_for_more = true;
                        }
                    } else if bar_buffer[n] == 2 {
                        if p.FFTbuffer_lower_cut_off[n - 1] + 1 < p.fft_midbuffer_size / 2 + 1 {
                            room_for_more = true;
                        }
                    } else if bar_buffer[n] == 3 {
                        if p.FFTbuffer_lower_cut_off[n - 1] + 1 < p.fft_treblebuffer_size / 2 + 1 {
                            room_for_more = true;
                        }
                    }
                    // dbg!(room_for_more);
                    if room_for_more {
                        p.FFTbuffer_lower_cut_off[n] = p.FFTbuffer_lower_cut_off[n - 1] + 1;
                        p.FFTbuffer_upper_cut_off[n - 1] = p.FFTbuffer_lower_cut_off[n] - 1;
                        if bar_buffer[n] == 1 {
                            relative_cut_off[n] = (p.FFTbuffer_lower_cut_off[n] as f32)
                                / (p.fft_bassbuffer_size as f32 / 2.0);
                        } else if bar_buffer[n] == 2 {
                            relative_cut_off[n] = (p.FFTbuffer_lower_cut_off[n] as f32)
                                / (p.fft_midbuffer_size as f32 / 2.0);
                        } else if bar_buffer[n] == 3 {
                            relative_cut_off[n] = p.FFTbuffer_lower_cut_off[n] as f32
                                / (p.fft_treblebuffer_size as f32 / 2.0);
                        }

                        p.cut_off_frequency[n] = relative_cut_off[n] as f32 * (p.rate as f32 / 2.0);
                        // dbg!(p.cut_off_frequency[n]);
                    }
                }
            } else {
                if p.FFTbuffer_upper_cut_off[n - 1] <= p.FFTbuffer_lower_cut_off[n - 1] {
                    p.FFTbuffer_upper_cut_off[n - 1] = p.FFTbuffer_lower_cut_off[n - 1] + 1;
                }
            }
        }
    }
    drop(bar_buffer);
    drop(relative_cut_off);

    return p;
}
fn execute(cava_in: &Vec<f32>, mut new_samples: usize, cava_out: &mut Vec<f32>, p: &mut Plan) {
    // overflow check
    if new_samples > p.input_buffer_size as usize {
        new_samples = p.input_buffer_size as usize;
    }
    // dbg!(p.input_buffer_size, new_samples);

    let mut silence: bool = true;
    if new_samples > 0 {
        p.framerate -= p.framerate / 64.0;
        p.framerate += (((p.rate * p.audio_channels) as i32 * p.frame_skip) as f32
            / new_samples as f32)
            / 64.0;
        p.frame_skip = 1;
        for n in (p.input_buffer_size - 1) as usize..new_samples {
            p.input_buffer[n] = p.input_buffer[n - new_samples];
        }
        // fill the input buffer
        for n in 0..new_samples {
            p.input_buffer[new_samples - n - 1] = cava_in[n];
            if cava_in[n] != 0.0 {
                silence = false;
            }
        }
    } else {
        p.frame_skip += 1;
    }
    // fill the bass, mid and treble buffers
    /* dbg!(
        p.fft_bassbuffer_size,
        p.in_bass_l_raw.len(),
        p.input_buffer.len()
    ); */
    for n in 0..p.fft_bassbuffer_size as usize {
        if p.audio_channels == 2 {
            p.in_bass_r_raw[n] = p.input_buffer[n * 2];
            p.in_bass_l_raw[n] = p.input_buffer[2 * n + 1];
        } else {
            p.in_bass_l_raw[n] = p.input_buffer[n];
        }
    }

    for n in 0..p.fft_midbuffer_size as usize {
        if p.audio_channels == 2 {
            p.in_mid_r_raw[n] = p.input_buffer[n * 2];
            p.in_mid_l_raw[n] = p.input_buffer[2 * n + 1];
        } else {
            p.in_mid_l_raw[n] = p.input_buffer[n];
        }
    }
    for n in 0..p.fft_treblebuffer_size as usize {
        if p.audio_channels == 2 {
            p.in_treble_r_raw[n] = p.input_buffer[n * 2];
            p.in_treble_l_raw[n] = p.input_buffer[2 * n + 1];
        } else {
            p.in_treble_l_raw[n] = p.input_buffer[n];
        }
    }
    // Hann Window
    for i in 0..p.fft_bassbuffer_size as usize {
        p.in_bass_l[i] = p.bass_multiplier[i] * p.in_bass_l_raw[i];
        if p.audio_channels == 2 {
            p.in_bass_r[i] = p.bass_multiplier[i] * p.in_bass_r_raw[i];
        }
    }

    for i in 0..p.fft_midbuffer_size as usize {
        p.in_mid_l[i] = p.mid_multiplier[i] * p.in_mid_l_raw[i];
        if p.audio_channels == 2 {
            p.in_mid_r[i] = p.mid_multiplier[i] * p.in_mid_r_raw[i];
        }
    }
    for i in 0..p.fft_treblebuffer_size as usize {
        p.in_treble_l[i] = p.treble_multiplier[i] * p.in_treble_l_raw[i];
        if p.audio_channels == 2 {
            p.in_treble_r[i] = p.treble_multiplier[i] * p.in_treble_r_raw[i];
        }
    }

    p.p_bass_l
        .as_mut()
        .unwrap()
        .r2c(&mut p.in_bass_l, &mut p.out_bass_l)
        .unwrap();
    /* for x in 0..p.out_bass_l.len() {
        print!("{:?}", p.out_bass_l[x]);
    } */
    p.p_mid_l
        .as_mut()
        .unwrap()
        .r2c(&mut p.in_mid_l, &mut p.out_mid_l)
        .unwrap();
    p.p_treble_l
        .as_mut()
        .unwrap()
        .r2c(&mut p.in_treble_l, &mut p.out_treble_l)
        .unwrap();
    if p.audio_channels == 2 {
        p.p_bass_r
            .as_mut()
            .unwrap()
            .r2c(&mut p.in_bass_r, &mut p.out_bass_r)
            .unwrap();
        p.p_mid_r
            .as_mut()
            .unwrap()
            .r2c(&mut p.in_mid_r, &mut p.out_mid_r)
            .unwrap();
        p.p_treble_r
            .as_mut()
            .unwrap()
            .r2c(&mut p.in_treble_r, &mut p.out_treble_r)
            .unwrap();
    }
    println!("Bass buffer l: ");
    for x in 0..10 {
        print!("{:?} ", p.out_bass_l[x]);
    }
    println!("");
    // separate frequency bands
    for n in 0..p.number_of_bars as usize {
        let mut temp_l: f32 = 0.0;
        let mut temp_r: f32 = 0.0;
        for i in p.FFTbuffer_lower_cut_off[n] as usize..=p.FFTbuffer_upper_cut_off[n] as usize {
            if n <= p.bass_cut_off_bar as usize {
                // dbg!(n,p.bass_cut_off_bar,hypot(p.out_bass_l[i].re, p.out_bass_l[i].im));
                temp_l += hypot(p.out_bass_l[i].re, p.out_bass_l[i].im);
                if p.audio_channels == 2 {
                    temp_r += hypot(p.out_bass_r[i].re, p.out_bass_r[i].im);
                }
            } else if n > p.bass_cut_off_bar as usize && n <= p.treble_cut_off_bar as usize {
                temp_l += hypot(p.out_mid_l[i].re, p.out_mid_l[i].im);
                if p.audio_channels == 2 {
                    temp_r += hypot(p.out_mid_r[i].re, p.out_mid_r[i].im);
                }
            } else if n > p.treble_cut_off_bar as usize {
                temp_l += hypot(p.out_treble_l[i].re, p.out_treble_l[i].im);
                if p.audio_channels == 2 {
                    temp_r += hypot(p.out_treble_r[i].re, p.out_treble_r[i].im);
                }
            }
        }
        // getting average multiply with eq
        temp_l /= (p.FFTbuffer_upper_cut_off[n] - p.FFTbuffer_lower_cut_off[n] + 1) as f32;
        temp_l *= p.eq[n];
        cava_out[n] = temp_l;

        if p.audio_channels == 2 {
            temp_r /= (p.FFTbuffer_upper_cut_off[n] - p.FFTbuffer_lower_cut_off[n] + 1) as f32;
            temp_r *= p.eq[n];
            cava_out[n + p.number_of_bars as usize] = temp_r;
        }
    }
    if p.autosens {
        for n in 0..(p.number_of_bars as u32 * p.audio_channels) as usize {
            cava_out[n as usize] *= p.sens;
        }
    }
    dbg!(&cava_out);
    let mut overshoot: bool = false;
    let mut gravity_mod: f32 = (60.0 / p.framerate).powf(2.5) * 1.54 / p.noise_reduction;
    if gravity_mod < 1.0 {
        gravity_mod = 1.0;
    }
    // dbg!(p.noise_reduction);
    for n in 0..(p.number_of_bars as u32 * p.audio_channels) as usize {
        // dbg!(p.prev_cava_out[n]);
        /* if n == 0 {
            dbg!(p.sens);
        } */
        if cava_out[n] < p.prev_cava_out[n] && p.noise_reduction > 0.1 {
            println!(
                "n: {} cava_out: {} prev_cava_out: {}",
                n, cava_out[n], p.prev_cava_out[n]
            );
            cava_out[n] = p.cava_peak[n] * (1.0 - (p.cava_fall[n] * p.cava_fall[n] * gravity_mod));
            if cava_out[n] < 0.0 {
                cava_out[n] = 0.0;
            }
            p.cava_fall[n] += 0.028;
        } else {
            p.cava_peak[n] = cava_out[n];
            p.cava_fall[n] = 0.0;
        }
        p.prev_cava_out[n] = cava_out[n];
        // process [smoothing]: integral
        cava_out[n] = p.cava_mem[n] * p.noise_reduction + cava_out[n];
        p.cava_mem[n] = cava_out[n];
        if p.autosens {
            // check if we overshoot target height
            if cava_out[n] > 1.0 {
                dbg!(n, cava_out[n]);
                overshoot = true;
            }
        }
    }
    println!("{:?}", cava_out);
    // calculating automatic sense adjustment
    println!(
        "{:?} {:?} {:?} {:?}",
        p.sens_init as u32, overshoot as u32, silence as u32, p.sens
    );
    if p.autosens {
        if overshoot {
            p.sens = p.sens * 0.98;
            p.sens_init = false;
        } else {
            if !silence {
                p.sens = p.sens * 1.002;
                if p.sens_init {
                    p.sens = p.sens * 1.1;
                }
            }
        }
    }
}

fn hypot(x: f32, y: f32) -> f32 {
    return (x * x + y * y).sqrt();
}

// test
#[cfg(test)]
mod tests {
    use std::f32::consts::PI;

    use crate::{execute, init};

    #[test]
    fn test_core() {
        let bars_per_channel = 10;
        let channels = 2;
        let buffer_size = 512 * channels; // number of samples per cava execute
        let rate = 44100;
        let noise_reduction = 0.77;
        let low_cut_off = 50;
        let high_cut_off = 10000;
        let blueprint_2000MHz = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.493, 0.446, 0.0, 0.0].to_vec();
        let blueprint_200MHz: Vec<_> = [0., 0., 0.978, 0.008, 0., 0.001, 0., 0., 0., 0.].to_vec();
        let mut plan = init(
            bars_per_channel,
            rate,
            channels,
            true,
            noise_reduction,
            low_cut_off,
            high_cut_off,
        );
        if plan.status < 0 {
            panic!("Error: {}\n", plan.error_message);
        }
        println!("got lower cut off frequecies");
        println!("{:#?}", plan.cut_off_frequency);
        println!("Sine wave test");
        let mut cava_out = vec![0_f32; (bars_per_channel as u32 * channels).try_into().unwrap()];
        let mut cava_in = vec![0_f32; buffer_size as usize];
        println!("Running execute 300 times (simulating 3.5 sec)");
        for k in 0..2 {
            for n in 0..(buffer_size / 2) as usize {
                cava_in[n * 2] = (2.0 * PI * 200.0 / rate as f32
                    * (n as u32 + (k * buffer_size / 2)) as f32)
                    .sin()
                    * 20000.0;
                cava_in[n * 2 + 1] = (2.0 * PI * 2000.0 / rate as f32
                    * (n as u32 + (k * buffer_size / 2)) as f32)
                    .sin()
                    * 20000.0;
            }
            println!("{:?}", cava_in);
            execute(&cava_in, buffer_size as usize, &mut cava_out, &mut plan);
        }
        // dbg!(&cava_out);

        for i in 0..(bars_per_channel * 2) as usize {
            cava_out[i] = (cava_out[i] * 1000.0).round() / 1000.0;
        }
        println!("Last output left, max value should be at 200Hz: ");
        println!("{:?}", cava_out);

        for i in 0..bars_per_channel as usize {
            if cava_out[i] > blueprint_200MHz[i] * 1.02 || cava_out[i] < blueprint_200MHz[i] * 0.98
            {
                panic!(
                    "Error: Value got:{:?}, Correct value: {:?}",
                    cava_out[i], blueprint_200MHz[i]
                );
            }
        }
        for i in 0..bars_per_channel as usize {
            if cava_out[i + bars_per_channel as usize] > blueprint_2000MHz[i] * 1.02
                || cava_out[i + bars_per_channel as usize] < blueprint_2000MHz[i] * 0.98
            {
                panic!(
                    "Error: Value got:{:?}, Correct value: {:?}",
                    cava_out[i], blueprint_2000MHz[i]
                );
            }
        }
        println!("All tests passed");
    }
}
