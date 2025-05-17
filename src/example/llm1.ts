import {pipeline,TextStreamer} from '@huggingface/transformers';

export async function main() {

    const generator = await pipeline(
        'text-generation',
        'onnx-community/DeepSeek-R1-Distill-Qwen-1.5B-ONNX',
        {dtype: "q4f16"},
    );

    // Define the list of messages
    const messages = [
        { role: "user", content:  "Solve the equation: x^2 - 3x + 2 = 0" },
    ];

// Create text streamer
    const streamer = new TextStreamer(generator.tokenizer, {
        skip_prompt: true,
        // callback_function: (text) => { }, // Optional callback function
    })

// Generate a response
    const output = await generator(messages, { max_new_tokens: 512, do_sample: false, streamer });
    //@ts-ignore
    console.log(output[0].generated_text.at(-1).content);

}

main()
    .then(() => process.exit(0))
    .catch(error => {
        console.error(error);
        process.exit(1);
    });
