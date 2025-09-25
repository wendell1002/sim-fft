#![no_std]
#![no_main]

use cortex_m::singleton;
use cortex_m_rt::entry;
use defmt::info;
use defmt_rtt as _;
use fugit::ExtU32 as _;
use fugit::Rate;
use panic_semihosting as _;

use sim_fft::{Complex, SimpleFft};
use stm32f1xx_hal::gpio::{self, Pin};
use stm32f1xx_hal::pac::ADC1;
use stm32f1xx_hal::rcc;
use stm32f1xx_hal::timer::{Channel, Tim2NoRemap};
use stm32f1xx_hal::{
    adc::{Adc, SampleTime},
    pac::{self},
    prelude::*,
    time::MonoTimer,
};

#[entry]
fn main() -> ! {
    //初始化和获取外设对象
    // 获取cortex-m 相关的核心外设
    let cp = cortex_m::Peripherals::take().unwrap();
    //获取stm32f1xx_hal硬件外设
    let dp = pac::Peripherals::take().unwrap();
    // 初始化并获取flash和rcc设备的所有权
    let mut flash = dp.FLASH.constrain();
    //冻结系统中所有时钟的配置，并将冻结后的频率值存储在“clocks”中
    // let clocks = rcc.cfgr.freeze(&mut flash.acr);
    info!("init");
    let sysclk = 56.MHz();
    let pclk = sysclk / 2;
    let mut rcc = dp.RCC.freeze(
        rcc::Config::hse(8.MHz())
            .sysclk(sysclk)
            .pclk1(pclk)
            .pclk2(sysclk)
            .hclk(sysclk)
            .adcclk(sysclk / 4),
        &mut flash.acr,
    );
    let clocks = rcc.clocks;

    info!(
        "sysclk:{:?},pclk1:{:?},pclk2:{:?},hclk:{:?},adcclk:{:?}",
        clocks.sysclk().to_Hz(),
        clocks.pclk1().to_Hz(),
        clocks.pclk2().to_Hz(),
        clocks.hclk().to_Hz(),
        clocks.adcclk().to_Hz()
    );

    let mut gpioa = dp.GPIOA.split(&mut rcc);
    let mut afio = dp.AFIO.constrain(&mut rcc);
    {
        let pa3 = gpioa.pa3.into_alternate_push_pull(&mut gpioa.crl);
        let mut pwm = dp
            .TIM2
            .pwm_hz::<Tim2NoRemap, _, _>(pa3, &mut afio.mapr, 100.kHz(), &mut rcc);
        pwm.enable(Channel::C4);
        pwm.set_period(136.kHz());
        pwm.set_duty(Channel::C4, pwm.get_max_duty() / 2);
    };

    let mut delay = dp.TIM3.delay_us(&mut rcc);

    // Setup ADC
    let mut adc1 = Adc::new(dp.ADC1, &mut rcc);
    adc1.set_sample_time(SampleTime::T_1);
    let pa1 = gpioa.pa1.into_analog(&mut gpioa.crl);
    let channes = dp.DMA1.split(&mut rcc);
    let mut adc1 = adc1.with_dma(pa1, channes.1);

    let mono = MonoTimer::new(cp.DWT, cp.DCB, &clocks);
    let frequency = mono.frequency();

    let mut buf = singleton!(ADC_BUFFER: [u16; ADC_BUFFER_SIZE] = [0; ADC_BUFFER_SIZE]).unwrap();

    loop {
        let now = mono.now();
        (buf, adc1) = adc1.read(buf).wait();
        let elapsed = now.elapsed() - 1250;
        let us = elapsed / (frequency.to_Hz() / 1_000_000_u32);
        let speed = buf.len() as u64 * 1000000 / us as u64;
        info!("elapsed: tick={},time={}us, speed={}hz", elapsed, us, speed);
        // info!("{:?}", buf);
        cal_hz(buf, speed as f64, (clocks.adcclk().to_Hz() / 14) as f64);
        delay.delay_ms(2000_u16);
    }
}

const ADC_BUFFER_SIZE: usize = 128;

fn cal_hz(buf: &mut [u16], clk1: f64, clk2: f64) -> usize {
    //大小为2^n
    let mut sin_table = [0.0; ADC_BUFFER_SIZE / 4 + 1];
    let mut data = [Complex::new(0.0, 0.0); ADC_BUFFER_SIZE];
    let mut fft = SimpleFft::new(&mut data, &mut sin_table, clk2 as f64);
    fft.init_sin_table();
    // 模拟 ADC 数据（u16）
    fft.load_data_u16(&buf);
    // 加窗（可选）
    fft.apply_kaiser_window(5.);
    // fft.apply_hamming_window();
    // 执行 FFT
    fft.fft();
    // info!("fft data {:?}", fft.data());

    // /获取主频
    let (freq1, _) = fft.dominant_frequency();
    let (freq2, mag) = fft.dominant_freq(clk1);
    let freq = (freq1 + freq2) / 2000.;
    info!("主频率:{}+{}= {} KHz, 幅度: {}", freq1, freq2, freq, mag);
    info!("peak val={}", fft.peak_val());

    // // 打印前10个频点
    // for i in 1..=10 {
    //     info!("f={}Hz, mag={}", fft.frequency(i), fft.magnitude(i));
    // }
    1
}
