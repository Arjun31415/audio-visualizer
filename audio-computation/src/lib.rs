use std::{cmp::min, f32::consts::PI};

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
        relative_cut_off[n] = (p.cut_off_frequency[n] / (p.rate as f32 / 2.0)) as i32;
        p.eq[n] = p.cut_off_frequency[n];
        p.eq[n] /= 2.0_f32.powi(29);
        p.eq[n] /= p.fft_bassbuffer_size.ilog2() as f32;
        if p.cut_off_frequency[n] < bass_cut_off as f32 {
            bar_buffer[n] = 1;
            p.FFTbuffer_lower_cut_off[n] = relative_cut_off[n] * (p.fft_bassbuffer_size / 2);
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
            p.FFTbuffer_lower_cut_off[n] = relative_cut_off[n] * (p.fft_midbuffer_size / 2);
            p.treble_cut_off_bar += 1;
            if p.treble_cut_off_bar - p.bass_cut_off_bar == 1 {
                first_bar = 1;
                if n > 0 {
                    p.FFTbuffer_upper_cut_off[n - 1] =
                        relative_cut_off[n] * (p.fft_bassbuffer_size / 2);
                }
            } else {
                first_bar = 0;
            }
            p.FFTbuffer_lower_cut_off[n] =
                min(p.FFTbuffer_lower_cut_off[n], p.fft_midbuffer_size / 2);
        } else {
            // TREBLE
            bar_buffer[n] = 3;
            p.FFTbuffer_lower_cut_off[n] = relative_cut_off[n] * (p.fft_treblebuffer_size / 2);
            first_treble_bar += 1;
            if first_treble_bar == 1 {
                first_bar = 1;
                if n > 0 {
                    p.FFTbuffer_upper_cut_off[n - 1] =
                        relative_cut_off[n] * (p.fft_midbuffer_size / 2);
                }
            } else {
                first_bar = 0;
            }
            p.FFTbuffer_lower_cut_off[n] =
                min(p.FFTbuffer_lower_cut_off[n], p.fft_treblebuffer_size / 2);
        }
        if n > 0 {
            if first_bar == 0 {
                p.FFTbuffer_upper_cut_off[n - 1] = p.FFTbuffer_lower_cut_off[n] - 1;
                // pushing the spectrum up if the exponential function gets "clumped" in the
                // bass and caluclating new cut off frequencies
                if p.FFTbuffer_lower_cut_off[n] <= p.FFTbuffer_lower_cut_off[n - 1] {
                    let mut room_for_more: i32 = 0;
                    if bar_buffer[n] == 1 {
                        if p.FFTbuffer_lower_cut_off[n - 1] + 1 < p.fft_bassbuffer_size / 2 + 1 {
                            room_for_more = 1;
                        }
                    } else if bar_buffer[n] == 2 {
                        if p.FFTbuffer_lower_cut_off[n - 1] + 1 < p.fft_midbuffer_size / 2 + 1 {
                            room_for_more = 1;
                        }
                    } else if bar_buffer[n] == 3 {
                        if p.FFTbuffer_lower_cut_off[n - 1] + 1 < p.fft_treblebuffer_size / 2 + 1 {
                            room_for_more = 1;
                        }
                    }
                    if room_for_more == 1 {
                        p.FFTbuffer_lower_cut_off[n] = p.FFTbuffer_lower_cut_off[n - 1] + 1;
                        p.FFTbuffer_upper_cut_off[n - 1] = p.FFTbuffer_lower_cut_off[n] - 1;
                        if bar_buffer[n] == 1 {
                            relative_cut_off[n] = ((p.FFTbuffer_lower_cut_off[n] as f32)
                                / (p.fft_bassbuffer_size as f32 / 2.0))
                                as i32;
                        } else if bar_buffer[n] == 2 {
                            relative_cut_off[n] = ((p.FFTbuffer_lower_cut_off[n] as f32)
                                / (p.fft_midbuffer_size as f32 / 2.0))
                                as i32;
                        } else if bar_buffer[n] == 3 {
                            relative_cut_off[n] = (p.FFTbuffer_lower_cut_off[n] as f32
                                / (p.fft_treblebuffer_size as f32 / 2.0))
                                as i32;
                        }

                        p.cut_off_frequency[n] = relative_cut_off[n] as f32 * (p.rate as f32 / 2.0);
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
fn execute(cava_in: Vec<f32>, cava_out: Vec<f32>, p: &mut Plan) {
    // overflow check
    let new_samples = cava_in.len();
    let mut silence: i32 = 1;
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
                silence = 0;
            }
        }
    } else {
        p.frame_skip += 1;
    }
    // fill the bass,mid and treble buffers
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
    p.p_bass_l
        .as_mut()
        .unwrap()
        .r2c(&mut p.in_bass_l, &mut p.out_bass_l)
        .unwrap();
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
    //separate frequency bands
}
