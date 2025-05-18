import ollama from 'ollama'
import {ChromaClient} from "chromadb";
import { pipeline } from '@huggingface/transformers';

export async function main() {


    const response = await ollama.chat({
        model: 'deepseek-r1:8b',
        messages: [
            {role: 'user', content: `啮齿类动物里面,小巧可爱的叫什么`},
            {role: 'assistant', content: `是仓鼠`},
            {role: 'user', content: `仓鼠最喜欢吃什么`}
        ],
    })

    console.log(response.message.role)
    console.log(response.message.content)

}

main()
    .then(() => process.exit(0))
    .catch(error => {
        console.error(error);
        process.exit(1);
    });
