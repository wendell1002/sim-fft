#![no_std]

use defmt::Format;
pub const PI: f64 = 3.14159265358979323846264338327950288419717_f64;
///
/// ```
/// 大小为2^n
/// const ADC_BUFFER_SIZE: usize = 128;
/// const SAMPLE_RATE: u32 = 1_000_000;
/// let sin_table = [0.0; ADC_BUFFER_SIZE / 4 + 1];
/// let data = [Complex::new(0.0, 0.0); ADC_BUFFER_SIZE];
/// let mut fft = SimpleFft::new(data, sin_table, SAMPLE_RATE as f64);
/// fft.init_sin_table();
/// // 模拟 ADC 数据（u16）
/// let buf = [
///     4094, 4095, 4095, 4095, 12, 11, 8, 8, 4094, 4095, 4095, 4095, 12, 8, 8, 11, 4094, 4095,
///     4095, 4095, 16, 8, 8, 9, 2236, 4095, 4095, 4095, 4095, 12, 11, 9, 10, 4095, 4095, 4095,
///     4095, 12, 11, 10, 11, 4094, 4095, 4095, 4095, 12, 10, 8, 10, 4095, 4095, 4095, 4095, 11,
///     10, 10, 10, 4094, 4095, 4095, 4095, 10, 10, 10, 10, 4095, 4095, 4095, 4095, 12, 11, 8, 8,
///     4094, 4095, 4095, 4095, 11, 9, 10, 12, 4094, 4095, 4095, 4095, 16, 11, 10, 10, 2233, 4095,
///     4095, 4095, 4095, 12, 10, 8, 10, 4095, 4095, 4095, 4095, 11, 9, 10, 11, 4094, 4095, 4095,
///     4095, 12, 8, 8, 10, 4095, 4095, 4095, 4095, 12, 8, 12, 8, 4094, 4095, 4095, 4095, 11, 10,
/// ];
/// fft.load_data_u16(&buf);
///     // 加窗（可选）
/// fft.apply_kaiser_window(8.5);
/// // 执行 FFT
/// fft.fft();
/// // /获取主频
/// let (freq, mag) = fft.dominant_frequency();
/// info!("主频率: {} Hz, 幅度: {}", freq, mag);
/// // 打印前10个频点
/// for i in 1..=10 {
///     info!("f={}Hz, mag={}", fft.frequency(i), fft.magnitude(i));
/// }
/// ```

#[derive(Clone, Copy, Debug, Format)]
pub struct Complex {
    real: f64,
    imag: f64,
}

impl Complex {
    pub fn new(real: f64, imag: f64) -> Self {
        Self { real, imag }
    }
}

///
/// 核心 FFT 处理器结构体
/// N FFT 点数
/// M 预计算正弦表：0 到 π/2，共 N/4 + 1 个点
pub struct SimpleFft<'a, const N: usize, const M: usize> {
    data: &'a mut [Complex; N],
    sin_table: &'a mut [f64; M],
    length: usize,
    sample_rate: f64,
}

impl<'a, const N: usize, const M: usize> SimpleFft<'a, N, M> {
    /// 创建一个新的 FFT 实例，指定点数 N，采样率 sample_rate
    /// # 参数
    /// - `data`: 输入数据数组，长度为 N
    /// - `sin_table`: 预计算正弦表，长度为 M = N/4 + 1 个点
    /// - `sample_rate`: 采样率，单位 Hz
    /// # 返回
    /// 新的 FFT 实例
    pub fn new(data: &'a mut [Complex; N], sin_table: &'a mut [f64; M], sample_rate: f64) -> Self {
        let size = data.len();
        Self {
            data,
            sin_table,
            length: size,
            sample_rate,
        }
    }

    ///初始化一遍，生成正弦表 创建一个正弦采样表，采样点数与福利叶变换点数相同
    /// 初始化正弦表，生成 0 到 π/2 的正弦值
    pub fn init_sin_table(&mut self) {
        for i in 0..=self.length / 4 {
            self.sin_table[i] = libm::sin(i as f64 * 2.0 * PI / self.length as f64);
        }
    }

    /// 查找正弦表中对应角度的正弦值
    /// # 参数
    /// - `x`: 角度，范围 [0, 1)
    /// # 返回
    /// 对应角度的正弦值
    fn sin_find(&mut self, x: f64) -> f64 {
        let mut i = (self.length as f64 * x) as usize;
        i >>= 1;
        if i > self.length / 4 {
            i = self.length / 2 - i;
        }
        self.sin_table[i]
    }
    /// 查找正弦表中对应角度的余弦值
    /// # 参数
    /// - `x`: 角度，范围 [0, 1)
    /// # 返回
    /// 对应角度的余弦值
    fn cos_find(&mut self, x: f64) -> f64 {
        let mut i = (self.length as f64 * x) as usize;
        i >>= 1;

        if i < self.length / 4 {
            self.sin_table[self.length / 4 - i]
        } else {
            -self.sin_table[i - self.length / 4]
        }
    }
    /// 变址
    /// 变址操作将数据数组中的元素重新排列，使得 FFT 可以在-place 进行计算
    fn change_seat(&mut self) {
        let next_value = self.length / 2;
        let next_m = self.length - 1;
        let mut j = 0;

        for i in 0..next_m {
            if i < j {
                self.data.swap(i, j);
            }
            let mut k = next_value;
            while k <= j {
                j -= k;
                k >>= 1;
            }
            j += k;
        }
    }
    /// FFT 运算函数
    /// 执行 FFT 变换，将输入数据从时域转换到频域
    pub fn fft(&mut self) {
        self.change_seat();
        let mut size = self.length;
        let mut m_of_n_fft = 1;
        while {
            size >>= 1;
            size != 1
        } {
            m_of_n_fft += 1;
        }

        for l in 1..=m_of_n_fft {
            let step = 1 << l;
            let b = step >> 1;
            for j in 0..b {
                let angle = j as f64 / b as f64;
                let w_real = self.cos_find(angle);
                let w_imag = -self.sin_find(angle);
                for k in (j..self.length).step_by(step) {
                    let kb = k + b;
                    let temp_xx_real = self.data[kb].real * w_real - self.data[kb].imag * w_imag;
                    let temp_xx_imag = w_imag * self.data[kb].real + self.data[kb].imag * w_real;

                    self.data[kb].real = self.data[k].real - temp_xx_real;
                    self.data[kb].imag = self.data[k].imag - temp_xx_imag;

                    self.data[k].real = self.data[k].real + temp_xx_real;
                    self.data[k].imag = self.data[k].imag + temp_xx_imag;
                }
            }
        }
    }

    /// 逆 FFT 运算函数
    pub fn ifft(&mut self) {
        self.change_seat();

        let mut l = self.length;
        let mut m_of_n_fft = 1;
        while {
            l >>= 1;
            l != 1
        } {
            m_of_n_fft += 1;
        }

        for l in 1..=m_of_n_fft {
            let step = 1 << l;
            let b = step >> 1;
            for j in 0..b {
                let angle = j as f64 / b as f64;
                let w_real = self.cos_find(angle);
                let w_imag = self.sin_find(angle);
                for k in (j..self.length).step_by(step) {
                    let kb = k + b;
                    let temp_xx_real = self.data[kb].real * w_real - self.data[kb].imag * w_imag;
                    let temp_xx_imag = w_imag * self.data[kb].real + self.data[kb].imag * w_real;

                    self.data[kb].real = self.data[k].real - temp_xx_real;
                    self.data[kb].imag = self.data[k].imag - temp_xx_imag;

                    self.data[k].real = self.data[k].real + temp_xx_real;
                    self.data[k].imag = self.data[k].imag + temp_xx_imag;
                }
            }
        }
    }

    /// 获取原始复数数据的不可变引用
    pub fn data(&self) -> &[Complex] {
        self.data
    }
    /// 从 f64 样本加载数据（实数信号）
    pub fn load_data_f64(&mut self, samples: &[f64]) {
        let n = self.length;
        for i in 0..n {
            self.data[i].real = samples[i % samples.len()];
            self.data[i].imag = 0.0;
            // self.data[i] = Complex::new(samples[i % samples.len()], 0.0);
        }
    }

    /// 从 u16 ADC 样本加载数据
    pub fn load_data_u16(&mut self, samples: &[u16]) {
        let n = self.length;
        for i in 0..n {
            self.data[i].real = samples[i % samples.len()] as f64;
            self.data[i].imag = 0.0;
            // self.data[i] = Complex::new(samples[i % samples.len()] as f64, 0.0);
        }
    }

    /// 获取第 i 个频点的幅度（归一化）,未验证准确
    fn magnitude_x(&self, i: usize) -> f64 {
        assert!(i < self.data.len());
        let mag = libm::sqrt(
            self.data[i].real * self.data[i].real + self.data[i].imag * self.data[i].imag,
        );
        let divisor = if i == 0 || i == self.data.len() / 2 {
            self.data.len() // DC 和 Nyquist 不乘 2
        } else {
            self.data.len() >> 1 // 其他频率：归一化为单边谱
        };
        mag / divisor as f64
    }
    /// 获取第 i 个频点复数的绝对值
    fn complex_abs(&self, i: usize) -> f64 {
        assert!(i < self.data.len());
        //取低十六位，虚部
        let l_x = self.data[i].imag;
        //取高十六位，实部
        let l_y = self.data[i].real;
        //除以32768再乘65536是为了符合浮点数计算规律,第 i 个频点复数的绝对值
        let x = self.data.len() as f64 * (l_x as f64) / 32768.0;
        let y = self.data.len() as f64 * (l_y as f64) / 32768.0;
        let mag = libm::sqrt(x * x + y * y) / self.data.len() as f64;
        let mag = (mag * 65536.0) as f64;
        // let mag = 65536.0
        //     * libm::sqrt(
        //         self.data[i].real * self.data[i].real / 32768.0
        //             + self.data[i].imag * self.data[i].imag / 32768.0,
        //     )
        //     / self.data.len() as f64;
        mag
    }

    /// 获取第 i 个频点复数的绝对值
    pub fn magnitude(&self, i: usize) -> f64 {
        let mag = self.complex_abs(i);
        let divisor = if i == 0 || i == self.data.len() / 2 {
            self.data.len() // DC 和 Nyquist 不乘 2
        } else {
            self.data.len() >> 1 // 其他频率：归一化为单边谱
        };
        mag / divisor as f64
    }
    /// 获取频点峰值
    pub fn peak_val(&self) -> ((usize, f64, f64), (usize, f64, f64)) {
        let mut max_mag = 0.0;
        let mut max_idx = 0;
        let mut second_max_mag = 0.0;
        let mut second_max_idx = 0;
        let n_half = self.data.len() / 2;
        // 遍历前一半频点，找到最大和次大的幅度
        //最大的通常为直流分量，次大的为非0hz频率分量峰值
        for i in 0..n_half {
            let mag = self.complex_abs(i);
            if mag > max_mag {
                second_max_mag = max_mag;
                second_max_idx = max_idx;
                max_mag = mag;
                max_idx = i;
            } else if mag > second_max_mag {
                second_max_mag = mag;
                second_max_idx = i;
            }
        }
        (
            (max_idx, max_mag, self.frequency(max_idx)),
            (
                second_max_idx,
                second_max_mag,
                self.frequency(second_max_idx),
            ),
        )
    }

    /// 获取第 i 个频点对应的频率（Hz）
    pub fn frequency(&self, i: usize) -> f64 {
        (i as f64 * self.sample_rate) / self.data.len() as f64
    }

    /// 获取最大幅度及其频率
    pub fn dominant_frequency(&self) -> (f64, f64) {
        let mut max_mag = 0.0;
        let mut max_idx = 0;
        let n_half = self.data.len() / 2;
        // 遍历前一半频点，找到最大和次大的幅度
        //最大的通常为直流分量，次大的为非0hz频率分量峰值
        for i in 0..n_half {
            let mag = self.magnitude(i);
            if mag > max_mag {
                max_mag = mag;
                max_idx = i;
            }
        }

        let freq = self.frequency(max_idx);
        (freq, max_mag)
    }
    /// 获取最大幅度及其频率
    pub fn dominant_freq(&self, sample_rate: f64) -> (f64, f64) {
        let mut max_mag = 0.0;
        let mut max_idx = 0;
        let n_half = self.data.len() / 2;
        // 遍历前一半频点，找到最大和次大的幅度
        //最大的通常为直流分量，次大的为非0hz频率分量峰值
        for i in 0..n_half {
            let mag = self.magnitude(i);
            if mag > max_mag {
                max_mag = mag;
                max_idx = i;
            }
        }

        let freq = (max_idx as f64 * sample_rate) / self.data.len() as f64;
        (freq, max_mag)
    }

    /// 应用汉明窗
    pub fn apply_hamming_window(&mut self) {
        let n = self.length;
        for i in 0..n {
            let window = 0.54 - 0.46 * libm::cos(2.0 * PI * i as f64 / (n as f64 - 1.0));
            self.data[i].real *= window;
        }
    }

    /// 应用汉宁窗（Hann）
    pub fn apply_hann_window(&mut self) {
        let n = self.length;
        for i in 0..n {
            let window = 0.5 * (1.0 - libm::cos(2.0 * PI * i as f64 / (n as f64 - 1.0)));
            self.data[i].real *= window;
        }
    }
    /// 应用布莱克曼窗
    pub fn apply_blackman_window(&mut self) {
        for n in 0..self.length {
            let w = 0.42 - 0.5 * libm::cos(2.0 * PI * n as f64 / (self.length as f64 - 1.0))
                + 0.08 * libm::cos(4.0 * PI * n as f64 / (self.length as f64 - 1.0));
            self.data[n].real *= w;
        }
    }

    /// 应用凯塞窗
    ///
    /// beta: 凯塞窗参数，控制窗口的形状, 控制窗函数旁瓣衰减的参数   形状参数（典型值：0 ~ 10）
    /// 当 beta 为 0 时，窗函数退化为矩形窗；
    /// 当 beta 为 5 时，窗函数退化为汉宁窗；
    /// 当 beta 为 10 时，窗函数退化为布莱克曼窗。
    pub fn apply_kaiser_window(&mut self, beta: f64) {
        let m = self.length - 1;
        let inv_i0_beta = 1.0 / self.i0_bessel(beta);
        for i in 0..self.length {
            let x = 2.0 * i as f64 / m as f64 - 1.0;
            let arg = beta * libm::sqrt(1.0 - libm::pow(x, 2.0));
            let w = self.i0_bessel(arg) * inv_i0_beta;
            self.data[i].real *= w;
        }
    }
    /// 修正的第一类零阶贝塞尔函数
    /// 零阶修正贝塞尔函数 I₀(x) 的近似计算
    /// 使用多项式展开（适用于 x < 3.75）和渐近展开（x >= 3.75）
    fn i0_bessel(&mut self, x: f64) -> f64 {
        let x_abs = x.abs();

        if x_abs < 3.75 {
            let y = libm::pow(x_abs / 3.75, 2.0);
            1.0 + 3.5156229 * y
                + 3.0899424 * libm::pow(y, 2.0)
                + 1.2067492 * libm::pow(y, 3.0)
                + 0.2659732 * libm::pow(y, 4.0)
                + 0.0360768 * libm::pow(y, 5.0)
                + 0.0045813 * libm::pow(y, 6.0)
        } else {
            let y = 3.75 / x_abs;
            (libm::exp(x_abs) / libm::sqrt(x_abs))
                * (0.39894228 + 0.01328592 * y + 0.00225319 * libm::pow(y, 2.0)
                    - 0.00157565 * libm::pow(y, 3.0)
                    + 0.00916281 * libm::pow(y, 4.0)
                    - 0.02057706 * libm::pow(y, 5.0)
                    + 0.02635537 * libm::pow(y, 6.0)
                    - 0.01647633 * libm::pow(y, 7.0)
                    + 0.00392377 * libm::pow(y, 8.0))
        }
    }
}
