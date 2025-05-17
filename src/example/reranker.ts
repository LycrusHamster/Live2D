import {AutoTokenizer, AutoModelForSequenceClassification} from '@huggingface/transformers';


export async function main() {

    let documents = [
        "The creator of Diablo II is Lycrus",
        "World of Warcraft is published on 2010",
        "完全不相干",
    ]

    let query = 'when does World of Warcraft publish'

    // const model_id = 'mixedbread-ai/mxbai-rerank-base-v1';
    const model_id = 'Xenova/bge-reranker-base';
    const model = await AutoModelForSequenceClassification.from_pretrained(model_id);
    const tokenizer = await AutoTokenizer.from_pretrained(model_id);


/*    const query = '什么是 RAG？';

    // 候选文档
    const documents = [
        "RAG 是一种结合检索与生成的大模型架构。",
        "向量数据库可以快速查找相关文档。",
        "深度学习在图像领域有广泛应用。"
    ];*/


    {
        //rerank demo
        let docs = documents
        let return_documents = true;
        let top_k = 10;

        const inputs = tokenizer(
            new Array(documents.length).fill(query),
            {
                text_pair: documents,
                padding: true,
                truncation: true,
            }
        )
        const {logits} = await model(inputs);

        console.log(`logits ${logits}`)

        let result = logits
            .sigmoid()
            .tolist()
            // @ts-ignore
            .map(([score], i) => ({
                corpus_id: i,
                score,
                ...(return_documents ? {text: documents[i]} : {})
            }))
            // @ts-ignore
            .sort((a, b) => b.score - a.score)
            .slice(0, top_k);


        console.log(`result ${result}`)

        // @ts-ignore
        result.forEach((element) => {console.log(`${element.score} : ${element.text}`)})
    }

}

main()
    .then(() => process.exit(0))
    .catch(error => {
        console.error(error);
        process.exit(1);
    });
