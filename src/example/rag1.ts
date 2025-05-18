import {ChromaClient} from "chromadb";
import {pipeline} from '@huggingface/transformers';

export async function main() {


    // Initialize the client
    const chroma = new ChromaClient({path: "http://localhost:8200"});

    // Create a collection
    // await chroma.deleteCollection({name: "my-collection"});
    const collection = await chroma.getOrCreateCollection({name: "my-collection"});


    const extractor = await pipeline('feature-extraction', 'Xenova/bge-base-zh');


    let documents = [
        "The creator of Diablo II is Lycrus",
        "World of Warcraft is published on 2010",
        "还有什么事",
        "2024年元旦是1月1日",
        "Diablo is a ARPG game"
    ]

    {
        {
            {


                for (let index in documents) {
                    let sentence = documents[index];

                    //mean?
                    const embeddings = await extractor(sentence, {pooling: 'mean', normalize: true});
                    // @ts-ignore
                    const embeddingData: Float32Array = embeddings[0].data
                    const embeddingDataNumberArray = [...embeddingData]
                    console.log(`embeddingDataNumberArray length ${embeddingDataNumberArray.length}`)

                    await collection.add({
                        embeddings: embeddingDataNumberArray,
                        ids: `lycrus-${index}`,
                        documents: sentence
                    })

                    console.log(`collection added for ${index}`)
                }
            }
        }
    }

    const countOfRag = await collection.count()
    console.log(`countOfRag: ${countOfRag}`)

    let query = 'when does Diablo publish'

    const embeddings = await extractor(query, {pooling: 'mean', normalize: true});
    // @ts-ignore
    const embeddingData: Float32Array = embeddings[0].data
    const embeddingDataNumberArray = [...embeddingData]
    console.log(`!!!embeddingDataNumberArray length ${embeddingDataNumberArray.length}`)


    let ragResult = await collection.query({
        queryEmbeddings: embeddingDataNumberArray,
        nResults: 10
    })

    let ragDocs = ragResult.documents[0];
    let ragDistances = ragResult.distances ? ragResult.distances[0] : [];

    let doc = ragDocs ?? `rag为空`

    {
        //rerank demo
        let docs = documents
        let results = [];

        const reranker = await pipeline('text-classification', 'Xenova/bge-reranker-base');

        for (const doc of docs) {
            const combined = `${query} </s> ${doc}`; // BGE Reranker 的输入格式
            const output = await reranker(combined, {top_k: 1}); // 返回单个分数

            //@ts-ignore
            console.log(`output ${output[0].score}`)

            //@ts-ignore
            results.push({doc, score: parseFloat(output[0].score)});
        }

        // 排序并打印
        results.sort((a, b) => b.score - a.score);

        console.log('排序结果：');
        results.forEach((r, i) => {
            console.log(`Top ${i + 1}: Score=${r.score.toFixed(4)}\n${r.doc}\n`);
        })
    }


    console.log(`doc: ${doc}`)

}

main()
    .then(() => process.exit(0))
    .catch(error => {
        console.error(error);
        process.exit(1);
    });
