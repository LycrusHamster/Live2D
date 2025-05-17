import {Injectable} from '@nestjs/common';
import {ChromaClient, Collection} from "chromadb";
import {FeatureExtractionPipeline, Pipeline, pipeline} from "@huggingface/transformers";
import {
    AutomaticSpeechRecognitionPipeline,
    TextClassificationPipeline,
    TextGenerationPipeline
} from "@huggingface/transformers/types/pipelines";
const sherpa_onnx = require('sherpa-onnx');

@Injectable()
export class HuggingFaceService {

    chroma: ChromaClient;
    collection: Collection;
    // embedding: FeatureExtractionPipeline;
    asr: AutomaticSpeechRecognitionPipeline;
    chat: TextGenerationPipeline;
    // reranker: TextClassificationPipeline;
    tts:any;

    constructor() {
        this.chroma = new ChromaClient({path: "http://localhost:8200"});
    }

    async onModuleInit() {
        this.collection = await this.chroma.getOrCreateCollection({name: "my-collection"});

/*
        //rag
        this.embedding = await pipeline('feature-extraction', 'Xenova/bge-base-zh');
*/

        console.log(`--------loading ASR Xenova/whisper-base`)
        //asr
        // @ts-ignore
        this.asr = await pipeline('automatic-speech-recognition', 'Xenova/whisper-base');


        console.log(`--------loading LLM onnx-community/Llama-3.2-1B-Instruct`)
        //llm
        this.chat = await pipeline('text-generation', 'onnx-community/Llama-3.2-1B-Instruct');

        //reranker
        /*this.reranker = await pipeline('text-classification', 'Xenova/bge-reranker-base');*/

        /*{
            //init rag

            let documents = [
                "小仓鼠最喜欢的食物是瓜子",
                "大龙猫最喜欢的食物是提摩西草",
            ]

            for (let index in documents) {
                let sentence = documents[index];

                //mean?
                const embeddings = await this.embedding(sentence, {pooling: 'mean', normalize: true});
                // @ts-ignore
                const embeddingData: Float32Array = embeddings[0].data
                const embeddingDataNumberArray = [...embeddingData]
                console.log(`embeddingDataNumberArray length ${embeddingDataNumberArray.length}`)

                await this.collection.add({
                    embeddings: embeddingDataNumberArray,
                    ids: `lycrus-${index}`,
                    documents: sentence
                })

                console.log(`collection added for ${index}`)
            }
        }*/


        console.log(`--------loading TTS sherpa_onnx`)
        this.tts = this.createOfflineTts();

        console.log(`--------onModuleInit done`)
    }

    onModuleDestroy() {
        this.tts.free();
        console.log('TTS model releases');
    }

    async handle(data: Float32Array):Promise<{
        asr: string,
        answer: string,
        audioData:Float32Array
    }> {

        //asr
        const asrResult = await this.asr(data, {
            // sampling_rate: 16000,
            language: `chinese`,
        });

        // @ts-ignore
        console.log(`asr: ${result.text}`);

        // @ts-ignore
        const text: string = asrResult.text;

        /*//rag
        const textEmbedding = await this.embedding(text, {pooling: 'mean', normalize: true});
        // @ts-ignore
        const textEmbeddingData: Float32Array = textEmbedding[0].data
        const textEmbeddingDataNumberArray = [...textEmbeddingData]
        console.log(`textEmbeddingDataNumberArray length ${textEmbeddingDataNumberArray.length}`)

        let ragResult = await this.collection.query({
            queryEmbeddings: textEmbeddingDataNumberArray,
            nResults: 1
        })




        //rerank
        let results = [];

        for (const doc of docs) {
            const combined = `${query} </s> ${doc}`; // BGE Reranker 的输入格式
            const output = await reranker(combined, {top_k: 1}); // 返回单个分数

            console.log(`output ${output}`)

            // results.push({ doc, score: parseFloat(output[0].score) });
        }

        // 排序并打印
        // results.sort((a, b) => b.score - a.score);
        //
        // console.log('排序结果：');
        // results.forEach((r, i) => {
        //     console.log(`Top ${i + 1}: Score=${r.score.toFixed(4)}\n${r.doc}\n`);
        // })
        */

        //chat
        const output = await this.chat(text, {
            // max_new_tokens: 512,
            temperature: 0.7,
            // stop: ['<|im_end|>'],
        });
        //@ts-ignore
        const answer = output[0].generated_text.replace(prompt, '').trim();

        //tts
        const speakerId = 66;
        const speed = 1.0;

        const audio = this.tts.generate(
            {text: text, sid: speakerId, speed: speed});

        const audioFloat32Array = audio.samples;

        return {
            asr: text,
            answer: answer,
            audioData:audioFloat32Array
        };
    }

    buildChatMLPrompt({system = [], user = [], assistant = [], rag = []}) {
        let prompt = '';

        if (rag.length) {
            const context = rag.map((item, i) => `${i + 1}. ${item}`).join('\n')

            prompt += `<|im_start|>system\nThe following context is for you to refer：\n${context}\n<|im_end|>\n`;

        }

        // 插入 system 提示
        for (const content of system) {
            prompt += `<|im_start|>system\n${content}<|im_end|>\n`;
        }

        // 将 user 与 assistant 按序拼接
        const maxTurns = Math.max(user.length, assistant.length);
        for (let i = 0; i < maxTurns; i++) {
            if (user[i]) {
                prompt += `<|im_start|>user\n${user[i]}<|im_end|>\n`;
            }
            if (assistant[i]) {
                prompt += `<|im_start|>assistant\n${assistant[i]}<|im_end|>\n`;
            }
        }

        // 结尾准备模型继续回复
        prompt += `<|im_start|>assistant\n`;

        return prompt;
    }

    createOfflineTts() {
        let offlineTtsVitsModelConfig = {
            model: './vits-icefall-zh-aishell3/model.onnx',
            lexicon: './vits-icefall-zh-aishell3/lexicon.txt',
            tokens: './vits-icefall-zh-aishell3/tokens.txt',
            noiseScale: 0.667,
            noiseScaleW: 0.8,
            lengthScale: 1.0,
        };
        let offlineTtsModelConfig = {
            offlineTtsVitsModelConfig: offlineTtsVitsModelConfig,
            numThreads: 1,
            debug: 1,
            provider: 'cpu',
        };

        let offlineTtsConfig = {
            offlineTtsModelConfig: offlineTtsModelConfig,
            ruleFsts:
                './vits-icefall-zh-aishell3/phone.fst,./vits-icefall-zh-aishell3/date.fst,./vits-icefall-zh-aishell3/number.fst,./vits-icefall-zh-aishell3/new_heteronym.fst',
            ruleFars: './vits-icefall-zh-aishell3/rule.far',
            maxNumSentences: 1,
        };

        return sherpa_onnx.createOfflineTts(offlineTtsConfig);
    }
}
