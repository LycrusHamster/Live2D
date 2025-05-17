import {pipeline} from '@huggingface/transformers';

export async function main() {

    const chat = await pipeline('text-generation', 'onnx-community/Llama-3.2-1B-Instruct');

    const query = `2025年中秋节是几号?`
    // const prompt = `<|im_start|>user\n${query}<|im_end|>`

    const output = await chat(query, {
        // max_new_tokens: 512,
        temperature: 0.7,
        // stop: ['<|im_end|>'],
    });

    //@ts-ignore
    // const answer = output[0].generated_text;
    const answer = output[0].generated_text.replace(prompt, '').trim();

    console.log(`answer: ${answer}`);
}

main()
    .then(() => process.exit(0))
    .catch(error => {
        console.error(error);
        process.exit(1);
    });
