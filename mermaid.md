graph TD
    A[Raw Text: "The meeting..."] --> B(文本清洗与归一化)
    B --> C[Normalized: "...two thirty p m..."]
    C --> D(G2P 字素转音素 - espeak-ng)
    D --> E[Phonemes: ð ə m iː t ɪ ŋ...]
    E --> F(ID 映射 & Interspersing)
    F --> G[Input Tensor: 1, 0, 15, 0, 22...]
    G --> H{ONNX Model Inference}
    I[Scales: 0.667, 1.0, 0.8] --> H
    J[Lengths: 85] --> H
    H --> K[Output Tensor: Float32 array]
    K --> L(音频反归一化 & Int16转换)
    L --> M[WAV Audio: Byte Stream]


这是一个非常好的实际案例。以 "The meeting is at 2:30 PM on Jan 5th." 这句话为例，我们可以清晰地看到数据如何在 Piper (基于 VITS) 的 Pipeline 中流转。
我们将流程分为三个主要阶段：预处理 (Preprocessing)、模型推理 (Inference) 和 后处理 (Post-processing)。
总体流程图
graph TD
    A --> B(文本清洗与归一化)
    B --> C[归一化文本:...two thirty p m...]
    C --> D(G2P 字素转音素 - espeak-ng)
    D --> E[音素序列: ð ə m iː t ɪ ŋ...]
    E --> F(ID 映射 & 填充)
    F --> G
    G --> H{ONNX 模型推理}
    I --> H
    J[Lengths: 85] --> H
    H --> K
    K --> L(音频反归一化 & Int16转换)
    L --> M

第一阶段：预处理 (Text to Tensor)
这一步的目标是将人类可读的文本转换为模型可读的整数张量。
1. 文本清洗与归一化 (Text Normalization)
神经网络模型无法直接理解 "2:30" 或 "Jan" 这种符号的读音，必须将其转换为发音单词。
 * 输入: "The meeting is at 2:30 PM on Jan 5th."
 * 处理: Piper 依赖 espeak-ng 或前端正则规则将数字、缩写、日期展开。
 * 输出 (中间数据):
   > "the meeting is at two thirty p m on january fifth"
   > (注：通常会转为小写，标点符号的处理取决于具体模型配置)
   > 
2. 字素转音素 (Phonemization - G2P)
将单词转换为国际音标 (IPA)。这是 Piper 声音自然的关键，它依赖 espeak-ng 引擎。
 * 输入: "the meeting is at two thirty p m on january fifth"
 * 处理: espeak-ng 查词典或使用发音规则预测发音。
 * 输出 (IPA 序列):
   > ð ə       m iː t ɪ ŋ       ɪ z       æ t       t uː       θ ɜː t i       p iː       ɛ m       ɒ n       d ʒ æ n j u ɛ r i       f ɪ f θ
   > (注：这里展示的是近似的美式英语 IPA，实际输出包含重音符号如 ˈ)
   > 
3. ID 映射与填充 (Token to ID & Interspersing)
模型只能接受数字 ID。这一步需要读取模型配套的 config.json 文件中的 phoneme_id_map。
 * 配置加载: 假设 phoneme_id_map 如下（仅为示例）：
   * ^ (BOS) = 1
   * $ (EOS) = 2
   * _ (PAD/Blank) = 0
   * ð = 15, ə = 16, m = 30, iː = 45...
 * 处理逻辑:
   * 映射: 遍历 IPA 序列，将每个字符转为 Int ID。
   * 添加特殊符: 句首加 BOS (1)，句尾加 EOS (2)。
   * Interspersing (插入空白): VITS 模型为了学习韵律对齐，通常需要在每两个音素之间插入一个 "空白" ID (通常是 0)。
 * 转换过程模拟:
   * 原始 ID 序列: [15, 16, 30, 45,...]
   * 插入空白后: [1, 0, 15, 0, 16, 0, 30, 0, 45, 0,..., 0, 2]
 * 输出 (最终 Input Tensor 数据):
   > [1, 0, 15, 0, 16, 0, 30, 0, 45, 0,..., 0, 2]
   > (假设这一长串 ID 数组长度为 N，例如 85)
   > 
第二阶段：模型推理 (ONNX Runtime)
此时，数据被送入 ONNX Runtime 引擎 (sess.run)。Piper 的 ONNX 模型通常需要以下四个输入张量：
1. 输入张量准备
| 张量名称 | 数据类型 | 形状 (Shape) | 示例数据 | 含义 |
|---|---|---|---|---|
| input | int64 | [1, N] | [[1, 0, 15, 0...]] | 音素 ID 序列 (Batch=1) |
| input_lengths | int64 | [1] | `` | 序列的总长度 |
 * 关于 scales 参数详解：
   * 0.667 (Noise Scale): 控制发音的随机性变化，值越大声音越富有感情但也越不稳定。
   * 1.0 (Length Scale): 语速控制。1.0 为原速，1.2 变慢，0.8 变快。
   * 0.8 (Noise W): 控制音素时长的随机性，影响韵律节奏。
2. 模型内部处理 (黑盒)
 * Text Encoder: 将输入的 ID 序列转换为高维向量特征。
 * Duration Predictor: 结合 ID 特征和 scales 参数，预测每个音素应该发音多少帧（例如，ID 15 ð 预测持续 0.1秒）。
 * Upsampling: 根据预测的时长，将特征向量复制扩展。
 * Decoder (HiFi-GAN): 将扩展后的特征通过反卷积层生成最终的波形数据。
3. 模型输出
 * Output Name: 通常为 output 或 audio。
 * Shape: 。例如 （假设生成 3 秒音频，采样率 22050Hz）。
 * Data Type: Float32。
 * 示例数据: 一个包含浮点数的数组，数值范围约在 -1.0 到 1.0 之间。
   > [[[0.0012, 0.0045, -0.0123, -0.0567,...]]]
   > 
第三阶段：后处理 (Audio Generation)
模型输出的是浮点数，但计算机声卡播放或保存 WAV 文件通常需要 16-bit PCM 整数。
1. 范围限制与缩放 (Clamping & Scaling)
 * 输入: 模型输出的 Float32 数组。
 * 逻辑:
   * Clamp: 防止数值溢出，将所有值强行限制在 [-1.0, 1.0] 之间。若有值是 1.2，强制变为 1.0。
   * Scale: 乘以 int16 的最大幅值 (32767)。
 * 计算示例:
   * 值 0.582 -> 0.582 * 32767 = 19070.394
   * 值 -0.012 -> -0.012 * 32767 = -393.204
2. 类型转换 (Float to Int16)
 * 将上一步的计算结果取整并强制转换为 16位有符号整数 (int16)。
 * 19070.394 -> 19070 (二进制: 01001010 01111110)
3. 二进制封装 (To WAV)
 * 流式播放: 这些 int16 字节流直接写入音频驱动（如 ALSA）。
 * 保存文件: 在数据前加上 44 字节的标准 WAV Header（包含采样率 22050Hz, 单声道, 16bit 等信息），组合成最终的 .wav 文件。
关键数据变化总结表
| 步骤 | 数据形态 | 示例内容片段 | 备注 |
|---|---|---|---|
| 用户输入 | String | "The meeting is at 2:30..." | 原始文本 |
| 归一化 | String | "...two thirty..." | 展开数字和缩写 |
| 音素化 | List | ['ð', 'ə', 'm',...] | IPA 音标列表 |
| 模型输入 | Tensor (Int64) | [1, 0, 15, 0, 16,...] | 映射为ID并填充空白 |
| 模型输出 | Tensor (Float32) | [0.02, -0.51, 0.33,...] | 原始波形幅值 |
| 最终音频 | Binary (Int16) | \x4E\x7A\x00\xF2... | PCM 音频流 |
