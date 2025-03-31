# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 17:21:31 2025

This file evaluates a rag model

@author: Shivaang
"""

from datasets import Dataset
import os
from dotenv import load_dotenv
from loguru import logger
from dataclasses import dataclass
from answer_retrieval import RAGPipeline
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    context_recall,
    context_precision,
    answer_relevancy,
    answer_correctness,
    answer_similarity
)
import matplotlib.pyplot as plt

@dataclass
class Configurations:
    pinecone_index = "ceadar-documents"
    pinecone_api_key = 'PINECONE_API_KEY'
    attention_paper_qa = [
        "What is the main innovation of the Transformer model?",
        "The main innovation of the Transformer model is that it relies entirely on self-attention mechanisms instead of recurrence or convolutions to draw global dependencies between input and output. It's the first sequence transduction model to dispense with recurrence and convolutions entirely.",
        "What are the two main advantages of the Transformer over RNN-based models?",
        "The two main advantages are: 1) The Transformer allows for significantly more parallelization during training, as it doesn't have the sequential computation constraint of RNNs, and 2) It can learn long-range dependencies more easily due to shorter path lengths between any two positions in the network.",
        "What is the architecture of the Transformer model?",
        "The Transformer follows an encoder-decoder architecture. The encoder consists of 6 identical layers, each with two sub-layers: a multi-head self-attention mechanism and a position-wise fully connected feed-forward network. The decoder also has 6 identical layers with three sub-layers: the two from the encoder plus a multi-head attention layer that attends over the encoder's output.",
        "What are the three types of attention used in the Transformer?",
        "The Transformer uses multi-head attention in three different ways: 1) Encoder self-attention, where each position in the encoder can attend to all positions in the previous encoder layer, 2) Encoder-decoder attention, where each decoder position can attend to all positions in the encoder output, and 3) Decoder self-attention, which allows each position in the decoder to attend to all previous positions in the decoder.",
        "What is scaled dot-product attention?",
        "Scaled dot-product attention computes the dot products of queries with keys, divides each by √dₖ (where dₖ is the dimension of keys), and applies a softmax function to obtain weights on the values. The scaling factor (1/√d_k) is used to counteract the effect of large dot products pushing the softmax into regions with extremely small gradients.",
        "What is multi-head attention and why is it beneficial?",
        "Multi-head attention projects queries, keys, and values h times with different learned linear projections, performs attention in parallel on each of these projected versions, and then concatenates and projects the results. This allows the model to jointly attend to information from different representation subspaces at different positions, which a single attention head would average out.",
        "How does the Transformer model handle the absence of recurrence for sequence ordering?",
        "Since the Transformer contains no recurrence, it uses positional encodings added to the input embeddings to provide information about the relative or absolute position of tokens in the sequence. The paper uses sine and cosine functions of different frequencies for this purpose.",
        "What were the BLEU scores achieved by the Transformer on the WMT 2014 English-to-German translation task?",
        "The base Transformer model achieved 27.3 BLEU points, and the bigger Transformer model achieved 28.4 BLEU points on the WMT 2014 English-to-German translation task, outperforming previous state-of-the-art models by more than 2.0 BLEU points.",
        "What is the computational complexity per layer for self-attention compared to recurrent layers?",
        "A self-attention layer has a computational complexity of O(n2· d) per layer, where n is the sequence length and d is the representation dimension. In contrast, a recurrent layer has a complexity of O(n · d^2). Self-attention is faster when n < d, which is often the case in modern NLP applications.",
        "What regularization techniques were used in training the Transformer?",
        "The Transformer used three regularization techniques: 1) Residual dropout, applied to the output of each sub-layer before it is added to the sub-layer input and normalized, as well as to the sums of embeddings and positional encodings, 2) Label smoothing of value 0.1, and 3) Beam search with a beam size of 4 and length penalty alpha = 0.6 for inference.",
        "How does the Transformer prevent decoder positions from attending to subsequent positions?",
        "The Transformer prevents decoder positions from attending to subsequent positions by modifying the self-attention sub-layer in the decoder to mask out (setting to −∞) all values in the input of the softmax which correspond to illegal connections. This masking, combined with the offset of output embeddings by one position, ensures predictions for position i can depend only on known outputs at positions less than i.",
        "What optimizer was used to train the Transformer?",
        "The Transformer was trained using the Adam optimizer with theta1 = 0.9, theta2 = 0.98, and epsilon = 10e^-9. The learning rate varied over training according to a formula that increased it linearly for the first 4000 training steps and then decreased it proportionally to the inverse square root of the step number.",
        "What is the purpose of the Feed-Forward Networks in the Transformer architecture?",
        "Each layer in the encoder and decoder contains a position-wise feed-forward network, applied to each position separately and identically. It consists of two linear transformations with a ReLU activation in between. This can be described as two convolutions with kernel size 1, and it allows the model to process each position's representations independently.",
        "How was the training time for the Transformer compared to previous models?",
        "The Transformer was significantly faster to train than previous architectures. The base model trained for 100,000 steps (about 12 hours) on 8 NVIDIA P100 GPUs, while the big model trained for 300,000 steps (3.5 days). This was a fraction of the training cost of previous state-of-the-art models.",
        "How did the Transformer perform on English constituency parsing?",
        "The Transformer performed surprisingly well on English constituency parsing without task-specific tuning. With WSJ-only training, it achieved 91.3 F1, and with semi-supervised training, it achieved 92.7 F1, outperforming previous models except for the Recurrent Neural Network Grammar.",
        "What dimensions were used for the base and big Transformer models?",
        "The base model used dmodel = 512, dff = 2048, h = 8 (attention heads), and dk = dv = 64. The big model used dmodel = 1024, dff = 4096, h = 16, and a higher dropout rate of 0.3 (compared to 0.1 for the base model).",
        "Why is the maximum path length important in sequence transduction networks?",
        "The maximum path length between any two input and output positions is important because it affects the ability to learn long-range dependencies. Shorter paths make it easier for the model to learn these dependencies. In the Transformer, this path length is constant (O(1)), compared to O(n) for RNNs and O(log k(n)) for convolutional architectures.",
        "What was the batching strategy used during training?",
        "Sentence pairs were batched together by approximate sequence length. Each training batch contained a set of sentence pairs containing approximately 25,000 source tokens and 25,000 target tokens.",
        "What was the effect of varying the number of attention heads on model performance?",
        "The experiments showed that both too few and too many attention heads hurt performance. A single-head attention model was 0.9 BLEU worse than the best setting (8 heads for the base model), but performance also dropped with too many heads (32 for the base model).",
    ]

   
def main():
    load_dotenv()
    cfg = Configurations()
    
    logger.info("Extracting question and answers")
    questions_list = [cfg.attention_paper_qa[index] for index in range(0, len(cfg.attention_paper_qa), 2)]
    ground_truths_list = [cfg.attention_paper_qa[index] for index in range(1, len(cfg.attention_paper_qa), 2)]
    
    
    answers_list = []
    contexts_list = []
    
    logger.info("Initialising rag pipeline")
    rag_pipeline = RAGPipeline(
        pinecone_api_key=os.getenv(cfg.pinecone_api_key),
        pinecone_index_name=cfg.pinecone_index,
    )
    
    logger.info("Making inferences on sample questions")
    # Inference
    for query in questions_list:
        print(query)
        answer, context = rag_pipeline.answer_question(str(query))
        answers_list.append(answer)
        contexts_list.append([context])
    
    
    # To dict
    data = {
        "question": questions_list,
        "answer": answers_list,
        "contexts": contexts_list,
        "reference": ground_truths_list
    }
    
    logger.info("Evaluating data")
    
    # Convert dict to dataset
    dataset = Dataset.from_dict(data)

    result = evaluate(
        dataset = dataset, 
        metrics=[
          context_precision,
          faithfulness,
          answer_relevancy,
          context_recall,
          answer_relevancy,
          answer_correctness,
          answer_similarity
        ],
    )
    
    logger.info("Saving data to file")    
    df = result.to_pandas()
    
    df[[
          "context_precision",
          "faithfulness",
          "answer_relevancy",
          "context_recall",
          "answer_relevancy",
          "answer_correctness",
          "semantic_similarity"
        ]].mean().plot(kind="bar", title="Rag Model Performance")
    
    plt.savefig("outputs/Rag_model_performance.png", bbox_inches='tight')

    df.to_csv("outputs/rag_model_performance.csv")


if __name__=="__main__":
    main()
