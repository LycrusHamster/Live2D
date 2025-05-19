import {Injectable, Logger} from '@nestjs/common';
import {ChromaClient, Collection} from "chromadb";
import {Chat, FeatureExtractionPipeline, Pipeline, pipeline} from "@huggingface/transformers";
import {
    AutomaticSpeechRecognitionPipeline,
    TextClassificationPipeline,
    TextGenerationPipeline
} from "@huggingface/transformers/types/pipelines";
import ollama, {Message} from 'ollama'
import {Cron, CronExpression} from "@nestjs/schedule";
import * as wavEncoder from 'wav-encoder';

const sherpa_onnx = require('sherpa-onnx');

@Injectable()
export class HuggingFaceService {

    private readonly logger = new Logger(HuggingFaceService.name);

    inited = false;
    chroma: ChromaClient;
    collection: Collection;
    embedding: FeatureExtractionPipeline;
    asr: AutomaticSpeechRecognitionPipeline;
    // chat: TextGenerationPipeline;
    // reranker: TextClassificationPipeline;
    tts: any;
    modelName = 'deepseek-r1:1.5b';

    constructor() {
        this.chroma = new ChromaClient({path: "http://localhost:8200"});
    }

    async onModuleInit() {
        /*console.log(`--------clear collection`)
        await this.chroma.deleteCollection({name: "my-collection"});*/

        this.collection = await this.chroma.getOrCreateCollection({name: "my-collection"});

        console.log(`--------loading RAG Xenova/bge-base-zh`)
        //rag
        // @ts-ignore
        this.embedding = await pipeline('feature-extraction', 'Xenova/bge-base-zh');

        console.log(`--------loading ASR Xenova/whisper-base`)
        //asr
        this.asr = await pipeline('automatic-speech-recognition', 'Xenova/whisper-base');


        /*console.log(`--------loading LLM onnx-community/Llama-3.2-1B-Instruct`)
        //llm
        this.chat = await pipeline('text-generation', 'onnx-community/Llama-3.2-1B-Instruct');*/

        //reranker
        /*this.reranker = await pipeline('text-classification', 'Xenova/bge-reranker-base');*/

        {
            //init rag

            const countOfRag = await this.collection.count()
            console.log(`countOfRag: ${countOfRag}`)
            /*if (countOfRag == 0) */
            {

                //hard code for now
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
            }
        }


        console.log(`--------loading TTS sherpa_onnx`)
        this.tts = this.createOfflineTts();

        await ollama.chat({
            model: this.modelName,
            messages: [{ role: 'user', content: 'ping' }],
            stream: false,
            keep_alive : 60*60
        });

        this.inited = true;

        console.log(`--------onModuleInit done`)
    }

    onModuleDestroy() {
        this.tts.free();
        console.log('TTS model releases');
    }
/*

    //不知道为什么cron没有触发
    @Cron("0/30 * * * * *")
    async handleCron() {
        console.log(`handleCron ${this.inited}`)
        if(!this.inited){
            return;
        }
        try {
            await ollama.chat({
                model: this.modelName,
                messages: [{ role: 'user', content: 'ping' }],
                stream: false,
            });
            console.log(`[keepalive] ping success to model: ${this.modelName}`);
        } catch (error) {
            console.log(`[keepalive] ping failed: ${error.message}`);
        }
    }
*/

    async handle(data: Float32Array): Promise<{
        asr: string,
        answer: string,
        wavFileDataBase64: string
    }> {

        //asr
        const asrResult = await this.asr(data, {
            // sampling_rate: 16000,
            language: `chinese`,
        });

        // @ts-ignore
        console.log(`asr: ${asrResult.text}`);

        // @ts-ignore
        let text: string = asrResult.text;

        text = `小仓鼠最喜欢的食物是什么`

        //rag
        const textEmbedding = await this.embedding(text, {pooling: 'mean', normalize: true});
        // @ts-ignore
        const textEmbeddingData: Float32Array = textEmbedding[0].data
        const textEmbeddingDataNumberArray = [...textEmbeddingData]
        console.log(`textEmbeddingDataNumberArray length ${textEmbeddingDataNumberArray.length}`)

        let ragResult = await this.collection.query({
            queryEmbeddings: textEmbeddingDataNumberArray,
            nResults: 10
        })

        let ragDocsRaw = ragResult.documents[0];
        let ragDocs = ragDocsRaw.filter((item): item is string => item !== null)


        for (let ragDoc of ragDocs) {
            console.log(`rag: ${ragDoc}`);
        }

        //rerank
        /*let results = [];

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
        /*let query = this.buildConcatenatedPrompt([], [text], [], ragDocs)
        console.log(`query: ${JSON.stringify(query, null, 2)}`)
        const output = await this.chat(query, {
            // max_new_tokens: 512,
            temperature: 0.7,
            // stop: ['<|im_end|>'],
        });
        //@ts-ignore
        // const answer = output[0].generated_text.replace(prompt, '').trim();
        const answer = output[0].generated_text;
        console.log(`answer: ${JSON.stringify(answer, null, 2)}`)
        console.log(`chat: ${answer}`);*/

        let query = this.buildChatMLPromptStruct([`你是一个简洁的助手，请直接回答问题，不要添加任何额外分析或说明。`], [text], [], ragDocs)
        console.log(`query: ${JSON.stringify(query, null, 2)}`)
        const response = await ollama.chat({
            model: this.modelName,
            messages: query,
        })
        const answer = this.stripThinkBlock(response.message.content)
        console.log(`chat: ${answer}`)

        //tts
        const speakerId = 66;
        const speed = 1.0;

        const audio = this.tts.generate(
            {text: answer, sid: speakerId, speed: speed});

        console.log(`tts got audio`)

        // const audioFloat32Array = audio.samples;
        // const audioRate = audio.sampleRate;

        const wavData = await wavEncoder.encode({
            sampleRate: audio.sampleRate,
            channelData: [audio.samples] // 单通道
        });

        const wavFileDataBase64 = Buffer.from(wavData).toString(`base64`)

        return {
            asr: text,
            answer: answer,
            wavFileDataBase64: wavFileDataBase64
        };
    }

    buildChatMLPromptStruct(system: Array<string> = [], user: Array<string> = [], assistant: Array<string> = [], rag: Array<string> = []) {
        const messages: Message[] = [];

        if (rag.length > 0) {
            messages.push({
                role: 'system',
                content: `以下是你可以参考的资料：\n${rag.map((x, i) => `${i + 1}. ${x}`).join('\n')}`
            });
        }

        // 插入其他 system 设定
        for (const sys of system) {
            messages.push({
                role: 'system',
                content: sys
            });
        }

        // 按顺序插入 user/assistant 对话（允许对话轮数不对等）
        const maxTurns = Math.max(user.length, assistant.length);
        for (let i = 0; i < maxTurns; i++) {
            if (user[i]) messages.push({role: 'user', content: user[i]});
            if (assistant[i]) messages.push({role: 'assistant', content: assistant[i]});
        }

        return messages;
    }

    buildChatMLPromptString(system: Array<string> = [], user: Array<string> = [], assistant: Array<string> = [], rag: Array<string> = []) {
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

    buildConcatenatedPrompt(system: Array<string> = [], user: Array<string> = [], assistant: Array<string> = [], rag: Array<string> = []) {
        let prompt = '';

        if (rag.length) {
            const context = rag.map((item, i) => `${i + 1}. ${item}`).join('\n')

            prompt += `The following context is for you to refer：\n${context}\n\n`;

        }

        // 插入 system 提示
        for (const content of system) {
            prompt += `  ${content}  \n`;
        }

        // 将 user 与 assistant 按序拼接
        const maxTurns = Math.max(user.length, assistant.length);
        for (let i = 0; i < maxTurns; i++) {
            if (user[i]) {
                prompt += `\n${user[i]}\n`;
            }
            if (assistant[i]) {
                prompt += `\n${assistant[i]}\n`;
            }
        }

        // 结尾准备模型继续回复
        prompt += `\n`;

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

    stripThinkBlock(text: string): string {
        // 去除 <think>...</think> 块（跨行、多段都可）
        return text.replace(/<think>[\s\S]*?<\/think>/g, '').trim();
    }
}
