import { pipeline } from '@huggingface/transformers';
import * as fs from 'fs';
import wav from 'wav-reader-ts';


export async function main() {

    // 加载 Whisper 模型（首次运行会下载）
    const transcriber = await pipeline('automatic-speech-recognition', 'Xenova/whisper-base');


// 读取 wav 文件
    const buffer = fs.readFileSync('/home/lycrus/Desktop/output.wav');
    const wavData = await wav.decode(buffer);

// 抽取 PCM 数据并转换为 Float32Array（Whisper 需要）
    const floatArray = new Float32Array(wavData.data.channelData[0].length);
    floatArray.set(wavData.data.channelData[0]);

// 推理（语音转文本）
    const result = await transcriber(floatArray, {
        // sampling_rate: 16000,
        language: `chinese`,
    });

    // @ts-ignore
    console.log(`asr: ${result.text}`);

}

main()
    .then(() => process.exit(0))
    .catch(error => {
        console.error(error);
        process.exit(1);
    });
