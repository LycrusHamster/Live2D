import process from "process";
import ollama from 'ollama'
import {ChromaClient} from "chromadb";
import { pipeline } from '@huggingface/transformers';

export async function main() {


    // Initialize the client
    const chroma = new ChromaClient({path: "http://localhost:8200"});

    // Create a collection
    const collection = await chroma.getOrCreateCollection({name: "my-collection"});

    let documents = [
        "The creator of Diablo II is Lycrus",
        "World of Warcraft is published on 2010",
    ]

    {{{



        for (let index in documents) {
            let sentence = documents[index];

            let res = await ollama.embed({
                model: 'mxbai-embed-large',
                input: sentence
            })

            console.log(`res: ${res.embeddings}`)

            await collection.add({
                embeddings: res.embeddings[0],
                ids: `lycrus-${index}`,
                documents: sentence
            })

            console.log(`collection added for ${index}`)
        }
    }}}


    let query = 'when does World of Warcraft publish'

    let embeddingQuery = await ollama.embed({
        model: 'mxbai-embed-large',
        input: query,
    })

    let ragResult = await collection.query({
        queryEmbeddings: embeddingQuery.embeddings,
        nResults: 1
    })

    {
        //rerank demo
        let docs = documents
        let results = [];

        const reranker = await pipeline('text-classification', 'Xenova/bge-reranker-base');

        for (const doc of docs) {
            const combined = `${query} </s> ${doc}`; // BGE Reranker 的输入格式
            const output = await reranker(combined, { top_k: 1 }); // 返回单个分数

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
    }

    let doc = ragResult.documents[0][0] ?? `rag为空`

    const response = await ollama.chat({
        model: 'deepseek-r1:8b',
        messages: [
            {role: 'assistant', content: doc},
            {role: 'user', content: query}
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
