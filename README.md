# awesome-code-generation-security


awesome code generation & security is a curated collection of scholarly papers and resources focusing on the nexus of code generation, testing, and security.  This repository aims to serve as a central hub for researchers, practitioners, and enthusiasts who are interested in exploring and advancing the state-of-the-art in code generation techniques and their implications for software security.

To stay updated with the latest additions and discussions, star or watch this repository.  Your involvement can make a significant difference in building a robust community around secure code generation!

> It is important to note that some papers are published in conferences or journals that do not offer public download resources. Additionally, some authors may not have made their work available on preprint servers like arXiv. For these resources, we will provide direct links to the official publication wherever possible. Since accessing these papers may require a subscription through academic institutions or individual purchases, we recommend that users contact their libraries or utilize academic networking resources to download them.

This warehouse directory group should be formed as follows:
- 1.Advancements in Code Generation Models: Techniques and Application
	- 1.1 Code Generation Models
	- 1.2 Code Generation Evaluation
	- 1.3 Others
- 2.Security in Code Generation: Approaches and Case Studies

>If you want to be involved in changes or have new resources you'd like to contribute to, you can make a request. 
> For more information, please contact us at thebinking66@gmail.com

# 1.Advancements in Code Generation Models: Techniques and Application

## 1.1 Code Generation Models

### 2019 

--------------------------------------------------------------------------------------------------------------------------
#### [ACL'19]  ERNIE-Code: Beyond English-Centric Cross-lingual Pretraining for

[[paper]](https://aclanthology.org/P19-1139.pdf)  [[project]](https://github.com/thunlp/ERNIE) 

<details>
  <summary>Click to see the abstract!</summary>
>  Neural language representation models such as BERT pre-trained on large-scale corpora can well capture rich semantic patterns from plain text, and be fine-tuned to consistently improve the performance of various NLP tasks. However, the existing pre-trained language models rarely consider incorporating knowledge graphs (KGs), which can provide rich structured knowledge facts for better language understanding. We argue that informative entities in KGs can enhance language representation with external knowledge. In this paper, we utilize both large-scale textual corpora and KGs to train an enhanced language representation model (ERNIE), which can take full advantage of lexical, syntactic, and knowledge information simultaneously. The experimental results have demonstrated that ERNIE achieves significant improvements on various knowledge-driven tasks, and meanwhile is comparable with the state-of-the-art model BERT on other common NLP tasks. The code and datasets will be available in the future.
</details>


### 2020 


--------------------------------------------------------------------------------------------------------------------------
#### [ESEC/FSE'20] IntelliCode Compose: Code Generation using Transformer

[[paper]](https://arxiv.org/pdf/2310.02059)  [[talk]](https://www.youtube.com/watch?v=sQpv_WY8n04)

<details>
  <summary>Click to see the abstract!</summary>
In software development through integrated development environments (IDEs), code completion is one of the most widely used features. Nevertheless, majority of integrated development environments only support completion of methods and APIs, or arguments.

In this paper, we introduce IntelliCode Compose – a general-purpose multilingual code completion tool which is capable of predicting sequences of code tokens of arbitrary types, generating up to entire lines of syntactically correct code. It leverages state-of-the-art generative transformer model trained on 1.2 billion lines of source code in Python, C#, JavaScript and TypeScript programming languages.  
IntelliCode Compose is deployed as a cloud-based web service. It makes use of client-side tree-based caching, efficient parallel implementation of the beam search decoder, and compute graph optimizations to meet edit-time completion suggestion requirements in the Visual Studio Code IDE and Azure Notebook.

Our best model yields an average edit similarity of 86.7% and a perplexity of 1.82 for Python programming language.
</details>



--------------------------------------------------------------------------------------------------------------------------
#### [AAAI'20]  ERNIE 2.0: A Continual Pre-Training Framework for Language Understanding


[[paper]](https://ojs.aaai.org/index.php/AAAI/article/download/6428/6284)  [[project]](https://github.com/thunlp/ERNIE) 

<details>
  <summary>Click to see the abstract!</summary>
>  Recently pre-trained models have achieved state-of-the-art results in various language understanding tasks. Current pre-training procedures usually focus on training the model with several simple tasks to grasp the co-occurrence of words or sentences. However, besides co-occurring information, there exists other valuable lexical, syntactic and semantic information in training corpora, such as named entities, semantic closeness and discourse relations. In order to extract the lexical, syntactic and semantic information from training corpora, we propose a continual pre-training framework named ERNIE 2.0 which incrementally builds pre-training tasks and then learn pre-trained models on these constructed tasks via continual multi-task learning. Based on this framework, we construct several tasks and train the ERNIE 2.0 model to capture lexical, syntactic and semantic aspects of information in the training data. Experimental results demonstrate that ERNIE 2.0 model outperforms BERT and XLNet on 16 tasks including English tasks on GLUE benchmarks and several similar tasks in Chinese. The source codes and pre-trained models have been released at https://github.com/PaddlePaddle/ERNIE.
</details>



### 2021


--------------------------------------------------------------------------------------------------------------------------
#### [arxiv'21] Evaluating Large Language Models Trained on Code
[[paper]](https://arxiv.org/pdf/2107.03374)  [[project]](https://github.com/openai/human-eval)  

<details>
  <summary>Click to see the abstract!</summary>
>  We introduce Codex, a GPT language model fine-tuned on publicly available code from GitHub, and study its Python code-writing capabilities. A distinct production version of Codex powers GitHub Copilot. On HumanEval, a new evaluation set we release to measure functional correctness for synthesizing programs from docstrings, our model solves 28.8% of the problems, while GPT-3 solves 0% and GPT-J solves 11.4%. Furthermore, we find that repeated sampling from the model is a surprisingly effective strategy for producing working solutions to difficult prompts. Using this method, we solve 70.2% of our problems with 100 samples per problem. Careful investigation of our model reveals its limitations, including difficulty with docstrings describing long chains of operations and with binding operations to variables. Finally, we discuss the potential broader impacts of deploying powerful code generation technologies, covering safety, security, and economics.
</details>

--------------------------------------------------------------------------------------------------------------------------
#### [ACL'21] CodeT5: Identifier-aware Unified Pre-trained Encoder-Decoder Models for Code Understanding and Generation
[[paper]](https://aclanthology.org/2021.emnlp-main.685.pdf)  [[project]](https://github.com/salesforce/CodeT5)  [[talk]](https://aclanthology.org/2021.emnlp-main.685.mp4)

<details>
  <summary>Click to see the abstract!</summary>
>  Pre-trained models for Natural Languages (NL) like BERT and GPT have been recently shown to transfer well to Programming Languages (PL) and largely benefit a broad set of code-related tasks. Despite their success, most current methods either rely on an encoder-only (or decoder-only) pre-training that is suboptimal for generation (resp. understanding) tasks or process the code snippet in the same way as NL, neglecting the special characteristics of PL such as token types. We present CodeT5, a unified pre-trained encoder-decoder Transformer model that better leverages the code semantics conveyed from the developer-assigned identifiers. Our model employs a unified framework to seamlessly support both code understanding and generation tasks and allows for multi-task learning. Besides, we propose a novel identifier-aware pre-training task that enables the model to distinguish which code tokens are identifiers and to recover them when they are masked. Furthermore, we propose to exploit the user-written code comments with a bimodal dual generation task for better NL-PL alignment. Comprehensive experiments show that CodeT5 significantly outperforms prior methods on understanding tasks such as code defect detection and clone detection, and generation tasks across various directions including PL-NL, NL-PL, and PL-PL. Further analysis reveals that our model can better capture semantic information from code. Our code and pre-trained models are released at https://github.com/salesforce/CodeT5.

</details>




--------------------------------------------------------------------------------------------------------------------------
#### [arxiv'21]  ERNIE 3.0: Large-scale Knowledge Enhanced Pre-training for Language Understanding and Generation

[[paper]](https://arxiv.org/pdf/2107.02137)  [[project]](https://github.com/thunlp/ERNIE) 

<details>
  <summary>Click to see the abstract!</summary>
>  > Pre-trained models have achieved state-of-the-art results in various Natural Language Processing (NLP) tasks. Recent works such as T5 and GPT-3 have shown that scaling up pre-trained language models can improve their generalization abilities. Particularly, the GPT-3 model with 175 billion parameters shows its strong task-agnostic zero-shot/few-shot learning capabilities. Despite their success, these large-scale models are trained on plain texts without introducing knowledge such as linguistic knowledge and world knowledge. In addition, most large-scale models are trained in an auto-regressive way. As a result, this kind of traditional fine-tuning approach demonstrates relatively weak performance when solving downstream language understanding tasks. In order to solve the above problems, we propose a unified framework named ERNIE 3.0 for pre-training large-scale knowledge enhanced models. It fuses auto-regressive network and auto-encoding network, so that the trained model can be easily tailored for both natural language understanding and generation tasks with zero-shot learning, few-shot learning or fine-tuning. We trained the model with 10 billion parameters on a 4TB corpus consisting of plain texts and a large-scale knowledge graph. Empirical results show that the model outperforms the state-of-the-art models on 54 Chinese NLP tasks, and its English version achieves the first place on the SuperGLUE benchmark (July 3, 2021), surpassing the human performance by +0.8% (90.6% vs. 89.8%).
</details>


### 2022 



--------------------------------------------------------------------------------------------------------------------------
#### [ACL'22] Efficient Large Scale Language Modeling with Mixtures of Experts

[[paper]](https://arxiv.org/pdf/2112.10684)  [[project]](https://github.com/facebookresearch/fairseq/tree/main/examples/moe_lm) 

<details>
  <summary>Click to see the abstract!</summary>
>  Mixture of Experts layers (MoEs) enable efficient scaling of language models through conditional computation. This paper presents a detailed empirical study of how autoregressive MoE language models scale in comparison with dense models in a wide range of settings: in- and out-of-domain language modeling, zero- and few-shot priming, and full-shot fine-tuning. With the exception of fine-tuning, we find MoEs to be substantially more compute efficient. At more modest training budgets, MoEs can match the performance of dense models using ~4 times less compute. This gap narrows at scale, but our largest MoE model (1.1T parameters) consistently outperforms a compute-equivalent dense model (6.7B parameters). Overall, this performance gap varies greatly across tasks and domains, suggesting that MoE and dense models generalize differently in ways that are worthy of future study. We make our code and models publicly available for research use.
</details>



--------------------------------------------------------------------------------------------------------------------------
#### [ACL'22] GPT-NeoX-20B: An Open-Source Autoregressive Language Model

[[paper]](https://aclanthology.org/2022.bigscience-1.9.pdf)  [[project]](https://github.com/eleutherai/gpt-neox) 


<details>
  <summary>Click to see the abstract!</summary>
>  We introduce GPT-NeoX-20B, a 20 billion parameter autoregressive language model trained on the Pile, whose weights will be made freely and openly available to the public through a permissive license. It is, to the best of our knowledge, the largest dense autoregressive model that has publicly available weights at the time of submission. In this work, we describe GPT-NeoX-20B’s architecture and training, and evaluate its performance. We open-source the training and evaluation code, as well as the model weights, at https://github.com/EleutherAI/gpt-neox.
</details>



--------------------------------------------------------------------------------------------------------------------------
#### [ACL'22] Evaluating Large Language Models Trained on Code

[[paper]](https://arxiv.org/pdf/2107.03374)  


<details>
  <summary>Click to see the abstract!</summary>
>  We introduce Codex, a GPT language model fine-tuned on publicly available code from GitHub, and study its Python code-writing capabilities. A distinct production version of Codex powers GitHub Copilot. On HumanEval, a new evaluation set we release to measure functional correctness for synthesizing programs from docstrings, our model solves 28.8% of the problems, while GPT-3 solves 0% and GPT-J solves 11.4%. Furthermore, we find that repeated sampling from the model is a surprisingly effective strategy for producing working solutions to difficult prompts. Using this method, we solve 70.2% of our problems with 100 samples per problem. Careful investigation of our model reveals its limitations, including difficulty with docstrings describing long chains of operations and with binding operations to variables. Finally, we discuss the potential broader impacts of deploying powerful code generation technologies, covering safety, security, and economics.

</details>




--------------------------------------------------------------------------------------------------------------------------
#### [arxiv'22] Efficient Training of Language Models to Fill in the Middle


[[paper]](https://arxiv.org/pdf/2207.14255) 

<details>
  <summary>Click to see the abstract!</summary>
>  We show that autoregressive language models can learn to infill text after we apply a straightforward transformation to the dataset, which simply moves a span of text from the middle of a document to its end. While this data augmentation has garnered much interest in recent years, we provide extensive evidence that training models with a large fraction of data transformed in this way does not harm the original left-to-right generative capability, as measured by perplexity and sampling evaluations across a wide range of scales. Given the usefulness, simplicity, and efficiency of training models to fill-in-the-middle (FIM), we suggest that future autoregressive language models be trained with FIM by default. To this end, we run a series of ablations on key hyperparameters, such as the data transformation frequency, the structure of the transformation, and the method of selecting the infill span. We use these ablations to prescribe strong default settings and best practices to train FIM models. We have released our best infilling model trained with best practices in our API, and release our infilling benchmarks to aid future research.
</details>



--------------------------------------------------------------------------------------------------------------------------
#### [Science'22] Competition-level code generation with AlphaCode

[[paper]](https://www.science.org/doi/epdf/10.1126/science.abq1158) [project](https://github.com/google-deepmind/code_contests)

<details>
  <summary>Click to see the abstract!</summary>
>  Programming is a powerful and ubiquitous problem-solving tool. Systems that can assist programmers or even generate programs themselves could make programming more productive and accessible. Recent transformer-based neural network models show impressive code generation abilities yet still perform poorly on more complex tasks requiring problem-solving skills, such as competitive programming problems. Here, we introduce AlphaCode, a system for code generation that achieved an average ranking in the top 54.3% in simulated evaluations on recent programming competitions on the Codeforces platform. AlphaCode solves problems by generating millions of diverse programs using specially trained transformer-based networks and then filtering and clustering those programs to a maximum of just 10 submissions. This result marks the first time an artificial intelligence system has performed competitively in programming competitions.
</details>






--------------------------------------------------------------------------------------------------------------------------
#### [ICLR'22 & PLDI'22]  A Systematic Evaluation of Large Language Models of Code



[[paper]](https://arxiv.org/pdf/2202.13169)  [[project]](https://github.com/VHellendoorn/Code-LMs) 

<details>
  <summary>Click to see the abstract!</summary>
> Large language models (LMs) of code have recently shown tremendous promise in completing code and synthesizing code from natural language descriptions. However, the current state-of-the-art code LMs (e.g., Codex) are not publicly available, leaving many questions about their model and data design decisions. We aim to fill in some of these blanks through a systematic evaluation of the largest existing models: Codex, GPT-J, GPT-Neo, GPT-NeoX-20B, and CodeParrot, across various programming languages. Although Codex itself is not open-source, we find that existing opensource models do achieve close results in some programming languages, although targeted mainly for natural language modeling. We further identify an important missing piece in the form of a large open-source model trained exclusively on a multi-lingual corpus of code. We release a new model, PolyCoder, with 2.7B parameters based on the GPT-2 architecture, that was trained on 249GB of code across 12 programming languages on a single machine. In the C programming language, PolyCoder outperforms all models including Codex. Our trained models are open-source and publicly available at https://github.com/VHellendoorn/Code-LMs, which enables future research and application in this area. We have an online appendix at https://arxiv.org/abs/2202.13169.
</details>



### 2023
--------------------------------------------------------------------------------------------------------------------------
#### [ICLR'23]  INCODER: A GENERATIVE MODEL FOR CODE INFILLING AND SYNTHESIS

[[paper]](https://arxiv.org/pdf/2204.05999)  [[project]](https://github.com/dpfried/incoder) [[slides]](https://secai.skku.edu/assets/paper-presentation/jiyong_InCoder_ICLR23.pdf) 

<details>
  <summary>Click to see the abstract!</summary>
>  Code is seldom written in a single left-to-right pass and is instead repeatedly edited and refined. We introduce InCoder, a unified generative model that can perform program synthesis (via left-to-right generation) as well as editing (via infilling). InCoder is trained to generate code files from a large corpus of permissively licensed code, where regions of code have been randomly masked and moved to the end of each file, allowing code infilling with bidirectional context. Our model is the first generative model that is able to directly perform zero-shot code infilling, which we evaluate on challenging tasks such as type inference, comment generation, and variable re-naming. We find that the ability to condition on bidirectional context substantially improves performance on these tasks, while still performing comparably on standard program synthesis benchmarks in comparison to left-to-right only models pretrained at similar scale. The InCoder models and code are publicly released. [this https URL](https://sites.google.com/view/incoder-code-models)
</details>



--------------------------------------------------------------------------------------------------------------------------
#### [arxiv'23]  CodeGen: An Open Large Language Model for Code with Multi-Turn Program Synthesis


[[paper]](https://arxiv.org/pdf/2203.13474)  [[project]](https://github.com/salesforce/CodeGen) 

<details>
  <summary>Click to see the abstract!</summary>
>  Program synthesis strives to generate a computer program as a solution to a given problem specification, expressed with input-output examples or natural language descriptions. The prevalence of large language models advances the state-of-the-art for program synthesis, though limited training resources and data impede open access to such models. To democratize this, we train and release a family of large language models up to 16.1B parameters, called CODEGEN, on natural language and programming language data, and open source the training library JAXFORMER. We show the utility of the trained model by demonstrating that it is competitive with the previous state-of-the-art on zero-shot Python code generation on HumanEval. We further investigate the multi-step paradigm for program synthesis, where a single program is factorized into multiple prompts specifying subproblems. To this end, we construct an open benchmark, Multi-Turn Programming Benchmark (MTPB), consisting of 115 diverse problem sets that are factorized into multi-turn prompts. Our analysis on MTPB shows that the same intent provided to CODEGEN in multi-turn fashion significantly improves program synthesis over that provided as a single turn. We make the training library JAXFORMER and model checkpoints available as open source contribution: this https URL.

</details>

--------------------------------------------------------------------------------------------------------------------------
#### [arxiv'23]  Outline, Then Details: Syntactically Guided Coarse-To-Fine Code Generation


[[paper]](https://arxiv.org/pdf/2305.00909)  [[project]](https://github.com/VITA-Group/ChainCoder) 

<details>
  <summary>Click to see the abstract!</summary>
>  For a complicated algorithm, its implementation by a human programmer usually starts with outlining a rough control flow followed by iterative enrichments, eventually yielding carefully generated syntactic structures and variables in a hierarchy. However, state-of-the-art large language models generate codes in a single pass, without intermediate warm-ups to reflect the structured thought process of "outline-then-detail". Inspired by the recent success of chain-of-thought prompting, we propose ChainCoder, a program synthesis language model that generates Python code progressively, i.e. from coarse to fine in multiple passes. We first decompose source code into layout frame components and accessory components via abstract syntax tree parsing to construct a hierarchical representation. We then reform our prediction target into a multi-pass objective, each pass generates a subsequence, which is concatenated in the hierarchy. Finally, a tailored transformer architecture is leveraged to jointly encode the natural language descriptions and syntactically aligned I/O data samples. Extensive evaluations show that ChainCoder outperforms state-of-the-arts, demonstrating that our progressive generation eases the reasoning procedure and guides the language model to generate higher-quality solutions. Our codes are available at: this https URL.
</details>





--------------------------------------------------------------------------------------------------------------------------
#### [arxiv'23]  CodeGeeX: A Pre-Trained Model for Code Generation with Multilingual Evaluations on HumanEval-X



[[paper]](https://arxiv.org/pdf/2303.17568)  [[project]](https://github.com/THUDM/CodeGeeX) 

<details>
  <summary>Click to see the abstract!</summary>
> Large pre-trained code generation models, such as OpenAI Codex, can generate syntax- and function-correct code, making the coding of programmers more productive and our pursuit of artificial general intelligence closer. In this paper, we introduce CodeGeeX, a multilingual model with 13 billion parameters for code generation. CodeGeeX is pre-trained on 850 billion tokens of 23 programming languages as of June 2022. Our extensive experiments suggest that CodeGeeX outperforms multilingual code models of similar scale for both the tasks of code generation and translation on HumanEval-X. Building upon HumanEval (Python only), we develop the HumanEval-X benchmark for evaluating multilingual models by hand-writing the solutions in C++, Java, JavaScript, and Go. In addition, we build CodeGeeX-based extensions on Visual Studio Code, JetBrains, and Cloud Studio, generating 4.7 billion tokens for tens of thousands of active users per week. Our user study demonstrates that CodeGeeX can help to increase coding efficiency for 83.4% of its users. Finally, CodeGeeX is publicly accessible and in Sep. 2022, we open-sourced its code, model weights (the version of 850B tokens), API, extensions, and HumanEval-X at this https URL.
</details>



--------------------------------------------------------------------------------------------------------------------------
#### [arxiv'23]  BLOOM: A 176B-Parameter Open-Access Multilingual Language Model



[[paper]](https://arxiv.org/pdf/2211.05100)  [[project]](https://huggingface.co/bigscience/bloom) 

<details>
  <summary>Click to see the abstract!</summary>
> Large language models (LLMs) have been shown to be able to perform new tasks based on a few demonstrations or natural language instructions. While these capabilities have led to widespread adoption, most LLMs are developed by resource-rich organizations and are frequently kept from the public. As a step towards democratizing this powerful technology, we present BLOOM, a 176B-parameter open-access language model designed and built thanks to a collaboration of hundreds of researchers. BLOOM is a decoder-only Transformer language model that was trained on the ROOTS corpus, a dataset comprising hundreds of sources in 46 natural and 13 programming languages (59 in total). We find that BLOOM achieves competitive performance on a wide variety of benchmarks, with stronger results after undergoing multitask prompted finetuning. To facilitate future research and applications using LLMs, we publicly release our models and code under the Responsible AI License.
</details>


--------------------------------------------------------------------------------------------------------------------------
#### [arxiv'23] PanGu-Coder2: Boosting Large Language Models for Code with Ranking Feedback
[[paper]](https://arxiv.org/pdf/2307.14936)  

<details>
  <summary>Click to see the abstract!</summary>
> Large Language Models for Code (Code LLM) are flourishing. New and powerful models are released on a weekly basis, demonstrating remarkable performance on the code generation task. Various approaches have been proposed to boost the code generation performance of pre-trained Code LLMs, such as supervised fine-tuning, instruction tuning, reinforcement learning, etc. In this paper, we propose a novel RRTF (Rank Responses to align Test&Teacher Feedback) framework, which can effectively and efficiently boost pre-trained large language models for code generation. Under this framework, we present PanGu-Coder2, which achieves 62.20% pass@1 on the OpenAI HumanEval benchmark. Furthermore, through an extensive evaluation on CoderEval and LeetCode benchmarks, we show that PanGu-Coder2 consistently outperforms all previous Code LLMs.
</details>


--------------------------------------------------------------------------------------------------------------------------
#### [arxiv'23] # MoTCoder: Elevating Large Language Models with Modular of Thought for Challenging Programming Tasks
[[paper]](https://arxiv.org/pdf/2312.15960)   [[project]](https://github.com/dvlab-research/MoTCoder)  

<details>
  <summary>Click to see the abstract!</summary>
> Large Language Models (LLMs) have showcased impressive capabilities in handling straightforward programming tasks. However, their performance tends to falter when confronted with more challenging programming problems. We observe that conventional models often generate solutions as monolithic code blocks, restricting their effectiveness in tackling intricate questions. To overcome this limitation, we present Modular-of-Thought Coder (MoTCoder). We introduce a pioneering framework for MoT instruction tuning, designed to promote the decomposition of tasks into logical sub-tasks and sub-modules. Our investigations reveal that, through the cultivation and utilization of sub-modules, MoTCoder significantly improves both the modularity and correctness of the generated solutions, leading to substantial relative pass@1 improvements of 12.9% on APPS and 9.43% on CodeContests. Our codes are available at this https URL.
</details>

--------------------------------------------------------------------------------------------------------------------------
#### [arxiv'23] Code Llama: Open Foundation Models for Code
[[paper]](https://arxiv.org/pdf/2308.12950)  [[project]](https://github.com/meta-llama/codellama)  

<details>
  <summary>Click to see the abstract!</summary>
>  We release Code Llama, a family of large language models for code based on Llama 2 providing state-of-the-art performance among open models, infilling capabilities, support for large input contexts, and zero-shot instruction following ability for programming tasks. We provide multiple flavors to cover a wide range of applications: foundation models (Code Llama), Python specializations (Code Llama - Python), and instruction-following models (Code Llama - Instruct) with 7B, 13B, 34B and 70B parameters each. All models are trained on sequences of 16k tokens and show improvements on inputs with up to 100k tokens. 7B, 13B and 70B Code Llama and Code Llama - Instruct variants support infilling based on surrounding content. Code Llama reaches state-of-the-art performance among open models on several code benchmarks, with scores of up to 67% and 65% on HumanEval and MBPP, respectively. Notably, Code Llama - Python 7B outperforms Llama 2 70B on HumanEval and MBPP, and all our models outperform every other publicly available model on MultiPL-E. We release Code Llama under a permissive license that allows for both research and commercial use.
</details>


--------------------------------------------------------------------------------------------------------------------------
#### [arxiv'23] CodeXGLUE: A Machine Learning Benchmark Dataset for Code Understanding and Generation
[[paper]](https://arxiv.org/pdf/2102.04664)  [[project]](https://github.com/microsoft/CodeXGLUE)  

<details>
  <summary>Click to see the abstract!</summary>
> Benchmark datasets have a significant impact on accelerating research in programming language tasks. In this paper, we introduce CodeXGLUE, a benchmark dataset to foster machine learning research for program understanding and generation. CodeXGLUE includes a collection of 10 tasks across 14 datasets and a platform for model evaluation and comparison. CodeXGLUE also features three baseline systems, including the BERT-style, GPT-style, and Encoder-Decoder models, to make it easy for researchers to use the platform. The availability of such data and baselines can help the development and validation of new methods that can be applied to various program understanding and generation problems.
</details>


--------------------------------------------------------------------------------------------------------------------------
#### [arxiv'23] SQL-PaLM: Improved large language model adaptation for Text-to-SQL (extended)
[[paper]](https://arxiv.org/pdf/2306.00739) 

<details>
  <summary>Click to see the abstract!</summary>
> Text-to-SQL, the process of translating natural language into Structured Query Language (SQL), represents a transformative application of large language models (LLMs), potentially revolutionizing how humans interact with data. This paper introduces the SQL-PaLM framework, a comprehensive solution for understanding and enhancing Text-to-SQL using LLMs, using in the learning regimes of few-shot prompting and instruction fine-tuning. With few-shot prompting, we explore the effectiveness of consistency decoding with execution-based error filtering. With instruction fine-tuning, we delve deep in understanding the critical paradigms that influence the performance of tuned LLMs. In particular, we investigate how performance can be improved through expanded training data coverage and diversity, synthetic data augmentation, and integrating query-specific database content. We propose a test-time selection method to further refine accuracy by integrating SQL outputs from multiple paradigms with execution feedback as guidance. Additionally, we tackle the practical challenge of navigating intricate databases with a significant number of tables and columns, proposing efficient techniques for accurately selecting relevant database elements to enhance Text-to-SQL performance. Our holistic approach yields substantial advancements in Text-to-SQL, as demonstrated on two key public benchmarks, Spider and BIRD. Through comprehensive ablations and error analyses, we shed light on the strengths and weaknesses of our framework, offering valuable insights into Text-to-SQL's future work.
</details>


--------------------------------------------------------------------------------------------------------------------------
#### [arxiv'23] CodeT5+: Open Code Large Language Models for Code Understanding and Generation
[[paper]](https://arxiv.org/pdf/2305.07922)  [[project]](https://github.com/salesforce/CodeT5/tree/main/CodeT5+)  

<details>
  <summary>Click to see the abstract!</summary>
>  Large language models (LLMs) pretrained on vast source code have achieved prominent progress in code intelligence. However, existing code LLMs have two main limitations in terms of architecture and pretraining tasks. First, they often adopt a specific architecture (encoder-only or decoder-only) or rely on a unified encoder-decoder network for different downstream tasks. The former paradigm is limited by inflexibility in applications while in the latter, the model is treated as a single system for all tasks, leading to suboptimal performance on a subset of tasks. Secondly, they often employ a limited set of pretraining objectives which might not be relevant to some downstream tasks and hence result in substantial performance degrade. To address these limitations, we propose ``CodeT5+'', a family of encoder-decoder LLMs for code in which component modules can be flexibly combined to suit a wide range of downstream code tasks. Such flexibility is enabled by our proposed mixture of pretraining objectives to mitigate the pretrain-finetune discrepancy. These objectives cover span denoising, contrastive learning, text-code matching, and causal LM pretraining tasks, on both unimodal and bimodal multilingual code corpora. Furthermore, we propose to initialize CodeT5+ with frozen off-the-shelf LLMs without training from scratch to efficiently scale up our models, and explore instruction-tuning to align with natural language instructions. We extensively evaluate CodeT5+ on over 20 code-related benchmarks in different settings, including zero-shot, finetuning, and instruction-tuning. We observe state-of-the-art (SoTA) model performance on various code-related tasks, such as code generation and completion, math programming, and text-to-code retrieval tasks. Particularly, our instruction-tuned CodeT5+ 16B achieves new SoTA results on HumanEval code generation task against other open code LLMs.
</details>


--------------------------------------------------------------------------------------------------------------------------
#### [OpenReview ICLR'23] CodeT5Mix: A Pretrained Mixture of Encoder-Decoder Transformers for Code Understanding and Generation
[[paper]](https://openreview.net/pdf?id=VPCi3STZcaO)  

<details>
  <summary>Click to see the abstract!</summary>
>  xxxxxxx
</details>





--------------------------------------------------------------------------------------------------------------------------
#### [arxiv'23] CodeT5+: Open Code Large Language Models for Code Understanding and Generation
[[paper]](https://arxiv.org/pdf/2305.07922)  [[project]](https://github.com/salesforce/CodeT5/tree/main/CodeT5+)  

<details>
  <summary>Click to see the abstract!</summary>
>  Large language models (LLMs) pretrained on vast source code have achieved prominent progress in code intelligence. However, existing code LLMs have two main limitations in terms of architecture and pretraining tasks. First, they often adopt a specific architecture (encoder-only or decoder-only) or rely on a unified encoder-decoder network for different downstream tasks. The former paradigm is limited by inflexibility in applications while in the latter, the model is treated as a single system for all tasks, leading to suboptimal performance on a subset of tasks. Secondly, they often employ a limited set of pretraining objectives which might not be relevant to some downstream tasks and hence result in substantial performance degrade. To address these limitations, we propose ``CodeT5+'', a family of encoder-decoder LLMs for code in which component modules can be flexibly combined to suit a wide range of downstream code tasks. Such flexibility is enabled by our proposed mixture of pretraining objectives to mitigate the pretrain-finetune discrepancy. These objectives cover span denoising, contrastive learning, text-code matching, and causal LM pretraining tasks, on both unimodal and bimodal multilingual code corpora. Furthermore, we propose to initialize CodeT5+ with frozen off-the-shelf LLMs without training from scratch to efficiently scale up our models, and explore instruction-tuning to align with natural language instructions. We extensively evaluate CodeT5+ on over 20 code-related benchmarks in different settings, including zero-shot, finetuning, and instruction-tuning. We observe state-of-the-art (SoTA) model performance on various code-related tasks, such as code generation and completion, math programming, and text-to-code retrieval tasks. Particularly, our instruction-tuned CodeT5+ 16B achieves new SoTA results on HumanEval code generation task against other open code LLMs.
</details>




## 1.2 Code Generation Evaluation


--------------------------------------------------------------------------------------------------------------------------
#### [NIPS'21] Measuring Coding Challenge Competence with APPS
[[paper]](https://arxiv.org/pdf/2105.09938)  [[project]](https://github.com/hendrycks/apps)  

<details>
  <summary>Click to see the abstract!</summary>
>  While programming is one of the most broadly applicable skills in modern society, modern machine learning models still cannot code solutions to basic problems. Despite its importance, there has been surprisingly little work on evaluating code generation, and it can be difficult to accurately assess code generation performance rigorously. To meet this challenge, we introduce APPS, a benchmark for code generation. Unlike prior work in more restricted settings, our benchmark measures the ability of models to take an arbitrary natural language specification and generate satisfactory Python code. Similar to how companies assess candidate software developers, we then evaluate models by checking their generated code on test cases. Our benchmark includes 10,000 problems, which range from having simple one-line solutions to being substantial algorithmic challenges. We fine-tune large language models on both GitHub and our training set, and we find that the prevalence of syntax errors is decreasing exponentially as models improve. Recent models such as GPT-Neo can pass approximately 20% of the test cases of introductory problems, so we find that machine learning models are now beginning to learn how to code. As the social significance of automatic code generation increases over the coming years, our benchmark can provide an important measure for tracking advancements.
</details>


--------------------------------------------------------------------------------------------------------------------------
#### [arXiv’23] Evaluating Instruction-Tuned Large Language Models on Code Comprehension and Generation
[[paper]](https://arxiv.org/abs/2308.01240)  

<details>
  <summary>Click to see the abstract!</summary>
>  In this work, we evaluate 10 open-source instructed LLMs on four representative code comprehension and generation tasks. We have the following main findings. First, for the zero-shot setting, instructed LLMs are very competitive on code comprehension and generation tasks and sometimes even better than small SOTA models specifically fine-tuned on each downstream task. We also find that larger instructed LLMs are not always better on code-related tasks. Second, for the few-shot setting, we find that adding demonstration examples substantially helps instructed LLMs perform better on most code comprehension and generation tasks; however, the examples would sometimes induce unstable or even worse performance. Furthermore, we find widely-used BM25-based shot selection strategy significantly outperforms the basic random selection or fixed selection only on generation problems. Third, for the fine-tuning setting, we find that fine-tuning could further improve the model performance on downstream code comprehension and generation tasks compared to the zero-shot/one-shot performance. In addition, after being fine-tuned on the same downstream task dataset, instructed LLMs outperform both the small SOTA models and similar-scaled LLMs without instruction tuning. Based on our findings, we further present practical implications on model and usage recommendation, performance and cost trade-offs, and future direction.
</details>

--------------------------------------------------------------------------------------------------------------------------
#### [ICLR'23] Multilingual Evaluation of Code Generation Models
[[paper]](https://arxiv.org/pdf/2210.14868)  [[project]](https://github.com/amazon-research/mxeval)  

<details>
  <summary>Click to see the abstract!</summary>
>  We present new benchmarks on evaluation code generation models: MBXP and Multilingual HumanEval, and MathQA-X. These datasets cover over 10 programming languages and are generated using a scalable conversion framework that transpiles prompts and test cases from the original Python datasets into the corresponding data in the target language. Using these benchmarks, we are able to assess the performance of code generation models in a multi-lingual fashion, and discovered generalization ability of language models on out-of-domain languages, advantages of multi-lingual models over mono-lingual, the ability of few-shot prompting to teach the model new languages, and zero-shot translation abilities even on mono-lingual settings. Furthermore, we use our code generation model to perform large-scale bootstrapping to obtain synthetic canonical solutions in several languages, which can be used for other code-related evaluations such as code insertion, robustness, or summarization tasks. Overall, our benchmarks represents a significant step towards a deeper understanding of language models' code generation abilities. We publicly release our code and datasets at [this https URL](https://github.com/amazon-research/mxeval).
</details>

--------------------------------------------------------------------------------------------------------------------------
#### [PMLR'23] DS-1000: A Natural and Reliable Benchmark for Data Science Code Generation
[[paper]](https://proceedings.mlr.press/v202/lai23b/lai23b.pdf)  [[project]](https://ds1000-code-gen.github.io/)  

<details>
  <summary>Click to see the abstract!</summary>
>  We introduce DS-1000, a code generation benchmark with a thousand data science problems spanning seven Python libraries, such as Numpy and Pandas. Compared to prior works, DS-1000 incorporates three core features. First, our problems reflect diverse, realistic, and practical use cases since we collected them from StackOverflow. Second, our automatic evaluation is highly specific (reliable) – across all Codex-002-predicted solutions that our evaluation accepts, only 1.8% of them are incorrect; we achieve this with multi-criteria metrics, checking both functional correctness by running test cases and surface-form constraints by restricting API usages or keywords. Finally, we proactively defend against memorization by slightly modifying our problems to be different from the original StackOverflow source; consequently, models cannot answer them correctly by memorizing the solutions from pre-training. The current best public system (Codex-002) achieves 43.3% accuracy, leaving ample room for improvement. We release our benchmark at https://ds1000-code-gen.github.io.
</details>

--------------------------------------------------------------------------------------------------------------------------
#### [arXiv'23] ClassEval: A Manually-Crafted Benchmark for Evaluating LLMs on Class-level Code Generation
[[paper]](https://arxiv.org/pdf/2308.01861)  [[project]]([https://github.com/FudanSELab/ClassEval](https://github.com/FudanSELab/ClassEval)) 

<details>
  <summary>Click to see the abstract!</summary>
>  In this work, we make the first attempt to evaluate LLMs in a more challenging code generation scenario, i.e. class-level code generation. We first manually construct the first class-level code generation benchmark ClassEval of 100 class-level Python code generation tasks with approximately 500 person-hours. Based on it, we then perform the first study of 11 state-of-the-art LLMs on class-level code generation. Based on our results, we have the following main findings. First, we find that all existing LLMs show much worse performance on class-level code generation compared to on standalone method-level code generation benchmarks like HumanEval; and the method-level coding ability cannot equivalently reflect the class-level coding ability among LLMs. Second, we find that GPT-4 and GPT-3.5 still exhibit dominate superior than other LLMs on class-level code generation, and the second-tier models includes Instruct-Starcoder, Instruct-Codegen, and Wizardcoder with very similar performance. Third, we find that generating the entire class all at once (i.e. holistic generation strategy) is the best generation strategy only for GPT-4 and GPT-3.5, while method-by-method generation (i.e. incremental and compositional) is better strategies for the other models with limited ability of understanding long instructions and utilizing the middle information. Lastly, we find the limited model ability of generating method-dependent code and discuss the frequent error types in generated classes. Our benchmark is available at [this https URL](https://github.com/FudanSELab/ClassEval).
</details>

--------------------------------------------------------------------------------------------------------------------------
#### [AAAI2024] Can LLM Replace Stack Overflow? A Study on Robustness and Reliability of Large Language Model Code Generation
[[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/30185)  [[project]](https://github.com/FloridSleeves/RobustAPI)  

<details>
  <summary>Click to see the abstract!</summary>
>  Recently, large language models (LLMs) have shown an extraordinary ability to understand natural language and generate programming code. It has been a common practice for software engineers to consult LLMs when encountering coding questions. Although efforts have been made to avoid syntax errors and align the code with the intended semantics, the reliability, and robustness of the code generation from LLMs have not yet been thoroughly studied. The executable code is not equivalent to reliable and robust code, especially in the context of real-world software development. For example, the misuse of APIs in the generated code could lead to severe problems, such as resource leaks, program crashes, etc. Existing code evaluation benchmarks and datasets focus on crafting small tasks such as programming questions in coding interviews, which, however, deviates from the problem that developers would ask LLM for real-world coding help. To fill the missing piece, in this work, we propose a dataset RobustAPI for evaluating the reliability and robustness of code generated by LLMs. We collect 1208 coding questions from Stack Overflow on 18 representative Java APIs. We summarize the common misuse patterns of these APIs and evaluate them on current popular LLMs. The evaluation results show that even for GPT-4, 62% of the generated code contains API misuses, which would cause unexpected consequences if the code is introduced into real-world software.
</details>

--------------------------------------------------------------------------------------------------------------------------
#### [NeurIPS 2023] Large Language Models of Code Fail at Completing Code with Potential Bugs
[[paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/819cebb05f993840e8a52d7564c5c282-Paper-Conference.pdf)  [[project]](https://github.com/amazon-science/buggy-code-completion)  

<details>
  <summary>Click to see the abstract!</summary>
>  Large language models of code (Code-LLMs) have recently brought tremendous advances to code completion, a fundamental feature of programming assistance and code intelligence. However, most existing works ignore the possible presence of bugs in the code context for generation, which are inevitable in software development. Therefore, we introduce and study the buggy-code completion problem, inspired by the realistic scenario of real-time code suggestion where the code context contains potential bugs – anti-patterns that can become bugs in the completed program. To systematically study the task, we introduce two datasets: one with synthetic bugs derived from semantics-altering operator changes (buggy-HumanEval) and one with realistic bugs derived from user submissions to coding problems (buggy-FixEval). We find that the presence of potential bugs significantly degrades the generation performance of the high-performing Code-LLMs. For instance, the passing rates of CODEGEN-2B-MONO on test cases of buggy-HumanEval drop more than 50% given a single potential bug in the context. Finally, we investigate several post-hoc methods for mitigating the adverse effect of potential bugs and find that there remains a large gap in post-mitigation performance.
</details>

--------------------------------------------------------------------------------------------------------------------------
#### [arXiv‘24] CodeApex: A Bilingual Programming Evaluation Benchmark for Large Language Models
[[paper]](https://arxiv.org/pdf/2309.01940)  [[project]]([https://apex.sjtu.edu.cn/codeapex/]()) 

<details>
  <summary>Click to see the abstract!</summary>
>  With the emergence of Large Language Models (LLMs), there has been a significant improvement in the programming capabilities of models, attracting growing attention from researchers. Evaluating the programming capabilities of LLMs is crucial as it reflects the multifaceted abilities of LLMs, and it has numerous downstream applications. In this paper, we propose CodeApex, a bilingual benchmark dataset focusing on the programming comprehension, code generation, and code correction abilities of LLMs. Programming comprehension task tests LLMs on multiple-choice exam questions covering conceptual understanding, commonsense reasoning, and multi-hop reasoning. The code generation task evaluates LLMs through completing C++ functions based on provided descriptions and prototypes. The code correction task asks LLMs to fix real-world erroneous code segments with different error messages. We evaluate 12 widely used LLMs, including both general-purpose and specialized models. GPT-4 exhibits the best programming capabilities, achieving approximate accuracy of 69%, 54%, and 66% on the three tasks, respectively. Compared to human performance, there is still significant room for improvement in LLM programming. We hope that CodeApex can serve as a reference for evaluating the coding capabilities of LLMs, further promoting their development and growth.
</details>

--------------------------------------------------------------------------------------------------------------------------
#### [ICSE'24] CoderEval: A Benchmark of Pragmatic Code Generation with Generative Pre-trained Models
[[paper]](https://dl.acm.org/doi/abs/10.1145/3597503.3623316)  [[project]](https://github.com/CoderEval/CoderEval) 

<details>
  <summary>Click to see the abstract!</summary>
>  Code generation models based on the pre-training and fine-tuning paradigm have been increasingly attempted by both academia and industry, resulting in well-known industrial models such as Codex, CodeGen, and PanGu-Coder. To evaluate the effectiveness of these models, multiple existing benchmarks (e.g., HumanEval and AiXBench) are proposed, including only cases of generating a standalone function, i.e., a function that may invoke or access only built-in functions and standard libraries. However, non-standalone functions, which typically are not included in the existing benchmarks, constitute more than 70% of the functions in popular open-source projects, and evaluating models' effectiveness on standalone functions cannot reflect these models' effectiveness on pragmatic code generation scenarios (i.e., code generation for real settings of open source or proprietary code).
To help bridge the preceding gap, in this paper, we propose a benchmark named CoderEval, consisting of 230 Python and 230 Java code generation tasks carefully curated from popular real-world open-source projects and a self-contained execution platform to automatically assess the functional correctness of generated code. CoderEval supports code generation tasks from six levels of context dependency, where context refers to code elements such as types, APIs, variables, and consts defined outside the function under generation but within the dependent third-party libraries, current class, file, or project. CoderEval can be used to evaluate the effectiveness of models in generating code beyond only standalone functions. By evaluating three state-of-the-art code generation models (CodeGen, PanGu-Coder, and ChatGPT) on CoderEval and HumanEval, we find that the effectiveness of these models in generating standalone functions is substantially higher than that in generating non-standalone functions. Our analysis highlights the current progress and pinpoints future directions to further improve a model's effectiveness by leveraging contextual information for pragmatic code generation.
</details>

--------------------------------------------------------------------------------------------------------------------------
#### [ICSE'24] Lost in Translation: A Study of Bugs Introduced by Large Language Models while Translating Code
[[paper]](https://dl.acm.org/doi/abs/10.1145/3597503.3639226)  [[project]](https://github.com/Intelligent-CAT-Lab/PLTranslationEmpirical) 

<details>
  <summary>Click to see the abstract!</summary>
>  Code translation aims to convert source code from one programming language (PL) to another. Given the promising abilities of large language models (LLMs) in code synthesis, researchers are exploring their potential to automate code translation. The prerequisite for advancing the state of LLM-based code translation is to understand their promises and limitations over existing techniques. To that end, we present a large-scale empirical study to investigate the ability of general LLMs and code LLMs for code translation across pairs of different languages, including C, C++, Go, Java, and Python. Our study, which involves the translation of 1,700 code samples from three benchmarks and two real-world projects, reveals that LLMs are yet to be reliably used to automate code translation---with correct translations ranging from 2.1% to 47.3% for the studied LLMs. Further manual investigation of _unsuccessful_ translations identifies 15 categories of translation bugs. We also compare LLM-based code translation with traditional non-LLM-based approaches. Our analysis shows that these two classes of techniques have their own strengths and weaknesses. Finally, insights from our study suggest that providing more context to LLMs during translation can help them produce better results. To that end, we propose a prompt-crafting approach based on the symptoms of erroneous translations; this improves the performance of LLM-based code translation by 5.5% on average. Our study is the first of its kind, in terms of scale and breadth, that provides insights into the current limitations of LLMs in code translation and opportunities for improving them. Our dataset---consisting of 1,700 code samples in five PLs with 10K+ tests, 43K+ translated code, 1,748 manually labeled bugs, and 1,365 bug-fix pairs---can help drive research in this area.
</details>


--------------------------------------------------------------------------------------------------------------------------
#### [arXiv‘24] BioCoder: A Benchmark for Bioinformatics Code Generation with Large Language Models
[[paper]](https://arxiv.org/pdf/2308.16458)  [[project]](https://github.com/gersteinlab/biocoder)  

<details>
  <summary>Click to see the abstract!</summary>
>  Pre-trained large language models (LLMs) have significantly improved code generation. As these models scale up, there is an increasing need for the output to handle more intricate tasks and to be appropriately specialized to particular domains. Here, we target bioinformatics due to the amount of domain knowledge, algorithms, and data operations this discipline requires. We present BioCoder, a benchmark developed to evaluate LLMs in generating bioinformatics-specific code. BioCoder spans much of the field, covering cross-file dependencies, class declarations, and global variables. It incorporates 1,026 Python functions and 1,243 Java methods extracted from GitHub, along with 253 examples from the Rosalind Project, all pertaining to bioinformatics. Using topic modeling, we show that the overall coverage of the included code is representative of the full spectrum of bioinformatics calculations. BioCoder incorporates a fuzz-testing framework for evaluation. We have applied it to evaluate various models including InCoder, CodeGen, CodeGen2, SantaCoder, StarCoder, StarCoder+, InstructCodeT5+, GPT-3.5, and GPT- 4. Furthermore, we fine-tuned one model (StarCoder), demonstrating that our training dataset can enhance the performance on our testing benchmark (by >15% in terms of Pass@K under certain prompt configurations and always >3%). The results highlight two key aspects of successful models: (1) Successful models accommodate a long prompt (> 2,600 tokens) with full context, including functional dependencies. (2) They contain domain-specific knowledge of bioinformatics, beyond just general coding capability. This is evident from the performance gain of GPT-3.5/4 compared to the smaller models on our benchmark (50% vs. up to 25%). Availability and implementation: Code is available at: [this https URL](https://github.com/gersteinlab/biocoder) and https://biocoder-benchmark. [this http URL](http://github.io/).
</details>

--------------------------------------------------------------------------------------------------------------------------
#### [ICML 2024] Language Agent Tree Search Unifies Reasoning, Acting, and Planning in Language Models
[[paper]](https://arxiv.org/pdf/2310.04406)  [[project]](https://github.com/lapisrocks/LanguageAgentTreeSearch)  

<details>
  <summary>Click to see the abstract!</summary>
>  While language models (LMs) have shown potential across a range of decision-making tasks, their reliance on simple acting processes limits their broad deployment as autonomous agents. In this paper, we introduce Language Agent Tree Search (LATS) -- the first general framework that synergizes the capabilities of LMs in reasoning, acting, and planning. By leveraging the in-context learning ability of LMs, we integrate Monte Carlo Tree Search into LATS to enable LMs as agents, along with LM-powered value functions and self-reflections for proficient exploration and enhanced decision-making. A key feature of our approach is the incorporation of an environment for external feedback, which offers a more deliberate and adaptive problem-solving mechanism that surpasses the constraints of existing techniques. Our experimental evaluation across diverse domains, including programming, interactive question-answering (QA), web navigation, and math, validates the effectiveness and generality of LATS in decision-making while maintaining competitive or improved reasoning performance. Notably, LATS achieves state-of-the-art pass@1 accuracy (92.7%) for programming on HumanEval with GPT-4 and demonstrates gradient-free performance (average score of 75.9) comparable to gradient-based fine-tuning for web navigation on WebShop with GPT-3.5. Code can be found at [this https URL](https://github.com/lapisrocks/LanguageAgentTreeSearch)
</details>

--------------------------------------------------------------------------------------------------------------------------
#### [arXiv’24] OctoPack: Instruction Tuning Code Large Language Models
[[paper]](https://arxiv.org/pdf/2308.07124)  [[project]](https://github.com/bigcode-project/octopack) 

<details>
  <summary>Click to see the abstract!</summary>
>  Finetuning large language models (LLMs) on instructions leads to vast performance improvements on natural language tasks. We apply instruction tuning using code, leveraging the natural structure of Git commits, which pair code changes with human instructions. We compile CommitPack: 4 terabytes of Git commits across 350 programming languages. We benchmark CommitPack against other natural and synthetic code instructions (xP3x, Self-Instruct, OASST) on the 16B parameter StarCoder model, and achieve state-of-the-art performance among models not trained on OpenAI outputs, on the HumanEval Python benchmark (46.2% pass@1). We further introduce HumanEvalPack, expanding the HumanEval benchmark to a total of 3 coding tasks (Code Repair, Code Explanation, Code Synthesis) across 6 languages (Python, JavaScript, Java, Go, C++, Rust). Our models, OctoCoder and OctoGeeX, achieve the best performance across HumanEvalPack among all permissive models, demonstrating CommitPack's benefits in generalizing to a wider set of languages and natural coding tasks. Code, models and data are freely available at [this https URL](https://github.com/bigcode-project/octopack).
</details>


--------------------------------------------------------------------------------------------------------------------------
#### [EACL'24]  ICE-Score: Instructing Large Language Models to Evaluate Code

[[paper]](https://arxiv.org/pdf/2304.14317)  [[project]](https://github.com/terryyz/ice-score) 

<details>
  <summary>Click to see the abstract!</summary>
>  Recent advancements in the field of natural language generation have facilitated the use of large language models to assess the quality of generated text. Although these models have shown promising results in tasks such as machine translation and summarization, their applicability in code intelligence tasks remains limited without human involvement. The complexity of programming concepts required for such tasks makes it difficult to develop evaluation metrics that align with human judgment. Token-matching-based metrics, such as BLEU, have demonstrated weak correlations with human practitioners in code intelligence tasks. Moreover, utilizing human-written test suites to evaluate functional correctness can be challenging in domains with low resources. To overcome these obstacles, we propose \texttt{ICE-Score}, a new evaluation metric via instructing large language models (LLMs) for code assessments. Our metric addresses the limitations of existing approaches by achieving superior correlations with functional correctness and human preferences, without the need for test oracles or references. We evaluate the efficacy of our metric on two different aspects (\textit{human preference} and \textit{execution success}) and four programming languages. Our results demonstrate that our metric surpasses state-of-the-art metrics for code generation, delivering high levels of accuracy and consistency across various programming languages and tasks. We also make our evaluation metric and datasets available to the public\footnote{\url{this https URL}}, encouraging further research in evaluating code intelligence tasks.
</details>



## 1.3 Others


--------------------------------------------------------------------------------------------------------------------------
#### [GITHUB] GPT-J

 [[project]](https://github.com/graphcore/gpt-j) 


<details>
  <summary>Click to see the intro!</summary>
>  GPT-J is an open-source alternative from EleutherAI to OpenAI's GPT-3. Available for anyone to download, GPT-J can be successfully fine-tuned to perform just as well as large models on a range of NLP tasks including question answering, sentiment analysis, and named entity recognition.

Try running GPT-J for yourself on Paperspace with Graphcore's IPU (Intelligence Processing Unit), a completely new kind of massively parallel processor to accelerate machine intelligence. Access advanced, cost-efficient IPU compute on demand in the cloud on Paperspace to build, fine-tune and deploy AI models such as GPT-J.
</details>


--------------------------------------------------------------------------------------------------------------------------
#### [arXiv’22] Large Language Models Meet NL2Code: A Survey
[[paper]](https://arxiv.org/abs/2212.09420)  [[project]](https://nl2code.github.io/) 

<details>
  <summary>Click to see the abstract!</summary>
>  The task of generating code from a natural language description, or NL2Code, is considered a pressing and significant challenge in code intelligence. Thanks to the rapid development of pre-training techniques, surging large language models are being proposed for code, sparking the advances in NL2Code. To facilitate further research and applications in this field, in this paper, we present a comprehensive survey of 27 existing large language models for NL2Code, and also review benchmarks and metrics. We provide an intuitive comparison of all existing models on the HumanEval benchmark. Through in-depth observation and analysis, we provide some insights and conclude that the key factors contributing to the success of large language models for NL2Code are "Large Size, Premium Data, Expert Tuning". In addition, we discuss challenges and opportunities regarding the gap between models and humans. We also create a website [this https URL](https://nl2code.github.io/) to track the latest progress through crowd-sourcing. To the best of our knowledge, this is the first survey of large language models for NL2Code, and we believe it will contribute to the ongoing development of the field.
</details>



--------------------------------------------------------------------------------------------------------------------------
#### [arXiv'23] OctoPack: Instruction Tuning Code Large Language Models
[[paper]](https://arxiv.org/abs/2308.07124)  [[project]](https://github.com/bigcode-project/octopack) 

<details>
  <summary>Click to see the abstract!</summary>
>  Finetuning large language models (LLMs) on instructions leads to vast performance improvements on natural language tasks. We apply instruction tuning using code, leveraging the natural structure of Git commits, which pair code changes with human instructions. We compile CommitPack: 4 terabytes of Git commits across 350 programming languages. We benchmark CommitPack against other natural and synthetic code instructions (xP3x, Self-Instruct, OASST) on the 16B parameter StarCoder model, and achieve state-of-the-art performance among models not trained on OpenAI outputs, on the HumanEval Python benchmark (46.2% pass@1). We further introduce HumanEvalPack, expanding the HumanEval benchmark to a total of 3 coding tasks (Code Repair, Code Explanation, Code Synthesis) across 6 languages (Python, JavaScript, Java, Go, C++, Rust). Our models, OctoCoder and OctoGeeX, achieve the best performance across HumanEvalPack among all permissive models, demonstrating CommitPack's benefits in generalizing to a wider set of languages and natural coding tasks. Code, models and data are freely available at [this https URL](https://github.com/bigcode-project/octopack).
</details>

--------------------------------------------------------------------------------------------------------------------------
#### [arXiv’23] Private-Library-Oriented Code Generation with Large Language Models
[[paper]](https://arxiv.org/abs/2307.15370)  

<details>
  <summary>Click to see the abstract!</summary>
>  Large language models (LLMs), such as Codex and GPT-4, have recently showcased their remarkable code generation abilities, facilitating a significant boost in coding efficiency. This paper will delve into utilizing LLMs for code generation in private libraries, as they are widely employed in everyday programming. Despite their remarkable capabilities, generating such private APIs poses a formidable conundrum for LLMs, as they inherently lack exposure to these private libraries during pre-training. To address this challenge, we propose a novel framework that emulates the process of programmers writing private code. This framework comprises two modules: APIFinder first retrieves potentially useful APIs from API documentation; and APICoder then leverages these retrieved APIs to generate private code. Specifically, APIFinder employs vector retrieval techniques and allows user involvement in the retrieval process. For APICoder, it can directly utilize off-the-shelf code generation models. To further cultivate explicit proficiency in invoking APIs from prompts, we continuously pre-train a reinforced version of APICoder, named CodeGenAPI. Our goal is to train the above two modules on vast public libraries, enabling generalization to private ones. Meanwhile, we create four private library benchmarks, including TorchDataEval, TorchDataComplexEval, MonkeyEval, and BeatNumEval, and meticulously handcraft test cases for each benchmark to support comprehensive evaluations. Numerous experiments on the four benchmarks consistently affirm the effectiveness of our approach. Furthermore, deeper analysis is also conducted to glean additional insights.
</details>

--------------------------------------------------------------------------------------------------------------------------
#### [arXiv’23] A Lightweight Framework for High-Quality Code Generation
[[paper]](https://arxiv.org/abs/2307.08220)  [[project]](https://zenodo.org/records/8015394)  

<details>
  <summary>Click to see the abstract!</summary>
>  In recent years, the use of automated source code generation utilizing transformer-based generative models has expanded, and these models can generate functional code according to the requirements of the developers. However, recent research revealed that these automatically generated source codes can contain vulnerabilities and other quality issues. Despite researchers' and practitioners' attempts to enhance code generation models, retraining and fine-tuning large language models is time-consuming and resource-intensive. Thus, we describe FRANC, a lightweight framework for recommending more secure and high-quality source code derived from transformer-based code generation models. FRANC includes a static filter to make the generated code compilable with heuristics and a quality-aware ranker to sort the code snippets based on a quality score. Moreover, the framework uses prompt engineering to fix persistent quality issues. We evaluated the framework with five Python and Java code generation models and six prompt datasets, including a newly created one in this work (SOEval). The static filter improves 9% to 46% Java suggestions and 10% to 43% Python suggestions regarding compilability. The average improvement over the NDCG@10 score for the ranking system is 0.0763, and the repairing techniques repair the highest 80% of prompts. FRANC takes, on average, 1.98 seconds for Java; for Python, it takes 0.08 seconds.
</details>

--------------------------------------------------------------------------------------------------------------------------
#### [arXiv’23] ToolCoder: Teach Code Generation Models to Use API Search Tools
[[paper]](https://arxiv.org/abs/2305.04032)  

<details>
  <summary>Click to see the abstract!</summary>
>  xAutomatically generating source code from natural language descriptions has been a growing field of research in recent years. However, current large-scale code generation models often encounter difficulties when selecting appropriate APIs for specific contexts. These models may generate APIs that do not meet requirements or refer to non-existent APIs in third-party libraries, especially for lesser-known or private libraries. Inspired by the process of human developers using tools to search APIs, we propose ToolCoder, a novel approach that integrates API search tools with existing models to assist in code generation and API selection. To teach our model to use tools, we introduce an automated data annotation method using ChatGPT to add tool usage information into the source code data and fine-tune code generation models. During inference, we integrate API search tools into the generation process so that our model can automatically use the search tool to get suggestions when selecting an API. Our experimental results demonstrate that ToolCoder exhibits excellent performance and generalization across five public and private library code generation benchmarks, with at least 6.21\% improvement on average pass@1 metrics and 9.64\% improvement on average pass@10 metrics compared to state-of-the-art methods. Furthermore, we show that our relatively small ToolCoder model is comparable to one of the current best models, GPT-3.5, highlighting the potential of incorporating programming tools into the code generation process.
</details>

--------------------------------------------------------------------------------------------------------------------------
#### [TSE 2023] An Empirical Evaluation of Using Large Language Models for Automated Unit Test Generation
[[paper]](https://ieeexplore.ieee.org/abstract/document/10329992)  [[project]](https://figshare.com/articles/dataset/An_Empirical_Evaluation_of_Using_Large_Language_Models_for_Automated_Unit_Test_Generation/23653371)  

<details>
  <summary>Click to see the abstract!</summary>
>  Unit tests play a key role in ensuring the correctness of software. However, manually creating unit tests is a laborious task, motivating the need for automation. Large Language Models (LLMs) have recently been applied to various aspects of software development, including their suggested use for automated generation of unit tests, but while requiring additional training or few-shot learning on examples of existing tests. This paper presents a large-scale empirical evaluation on the effectiveness of LLMs for automated unit test generation without requiring additional training or manual effort. Concretely, we consider an approach where the LLM is provided with prompts that include the signature and implementation of a function under test, along with usage examples extracted from documentation. Furthermore, if a generated test fails, our approach attempts to generate a new test that fixes the problem by re-prompting the model with the failing test and error message. We implement our approach in TestPilot , an adaptive LLM-based test generation tool for JavaScript that automatically generates unit tests for the methods in a given project's API. We evaluate TestPilot using OpenAI's _gpt3.5-turbo_ LLM on 25 npm packages with a total of 1,684 API functions. The generated tests achieve a median statement coverage of 70.2% and branch coverage of 52.8%. In contrast, the state-of-the feedback-directed JavaScript test generation technique, Nessie, achieves only 51.3% statement coverage and 25.6% branch coverage. Furthermore, experiments with excluding parts of the information included in the prompts show that all components contribute towards the generation of effective test suites. We also find that 92.8% of TestPilot 's generated tests have $\leq$ 50% similarity with existing tests (as measured by normalized edit distance), with none of them being exact copies. Finally, we run TestPilot with two additional LLMs, OpenAI's older _code-cushman-002_ LLM and _StarCoder_ , an LLM for which the training process is publicly documented. Overall, we observed similar results with the former (68.2% median statement coverage), and somewhat worse results with the latter (54.0% median statement coverage), suggesting that the effectiveness of the approach is influenced by the size and training set of the LLM, but does not fundamentally depend on the specific model.
</details>

--------------------------------------------------------------------------------------------------------------------------
#### [arXiv’23] Exploring the Robustness of Large Language Models for Solving Programming Problems
[[paper]](https://arxiv.org/abs/2306.14583) 

<details>
  <summary>Click to see the abstract!</summary>
>  Using large language models (LLMs) for source code has recently gained attention. LLMs, such as Transformer-based models like Codex and ChatGPT, have been shown to be highly capable of solving a wide range of programming problems. However, the extent to which LLMs understand problem descriptions and generate programs accordingly or just retrieve source code from the most relevant problem in training data based on superficial cues has not been discovered yet. To explore this research question, we conduct experiments to understand the robustness of several popular LLMs, CodeGen and GPT-3.5 series models, capable of tackling code generation tasks in introductory programming problems. Our experimental results show that CodeGen and Codex are sensitive to the superficial modifications of problem descriptions and significantly impact code generation performance. Furthermore, we observe that Codex relies on variable names, as randomized variables decrease the solved rate significantly. However, the state-of-the-art (SOTA) models, such as InstructGPT and ChatGPT, show higher robustness to superficial modifications and have an outstanding capability for solving programming problems. This highlights the fact that slight modifications to the prompts given to the LLMs can greatly affect code generation performance, and careful formatting of prompts is essential for high-quality code generation, while the SOTA models are becoming more robust to perturbations.
</details>


--------------------------------------------------------------------------------------------------------------------------
#### [arXiv’23] Automatic Code Summarization via ChatGPT: How Far Are We?
[[paper]](https://arxiv.org/abs/2305.12865) 

<details>
  <summary>Click to see the abstract!</summary>
>  To support software developers in understanding and maintaining programs, various automatic code summarization techniques have been proposed to generate a concise natural language comment for a given code snippet. Recently, the emergence of large language models (LLMs) has led to a great boost in the performance of natural language processing tasks. Among them, ChatGPT is the most popular one which has attracted wide attention from the software engineering community. However, it still remains unclear how ChatGPT performs in (automatic) code summarization. Therefore, in this paper, we focus on evaluating ChatGPT on a widely-used Python dataset called CSN-Python and comparing it with several state-of-the-art (SOTA) code summarization models. Specifically, we first explore an appropriate prompt to guide ChatGPT to generate in-distribution comments. Then, we use such a prompt to ask ChatGPT to generate comments for all code snippets in the CSN-Python test set. We adopt three widely-used metrics (including BLEU, METEOR, and ROUGE-L) to measure the quality of the comments generated by ChatGPT and SOTA models (including NCS, CodeBERT, and CodeT5). The experimental results show that in terms of BLEU and ROUGE-L, ChatGPT's code summarization performance is significantly worse than all three SOTA models. We also present some cases and discuss the advantages and disadvantages of ChatGPT in code summarization. Based on the findings, we outline several open challenges and opportunities in ChatGPT-based code summarization.
</details>



--------------------------------------------------------------------------------------------------------------------------
#### [ICSE’22] Jigsaw: large language models meet program synthesis
[[paper]](https://dl.acm.org/doi/abs/10.1145/3510003.3510203) 

<details>
  <summary>Click to see the abstract!</summary>
>  Large pre-trained language models such as GPT-3 [10], Codex [11], and Google's language model [7] are now capable of generating code from natural language specifications of programmer intent. We view these developments with a mixture of optimism and caution. On the optimistic side, such large language models have the potential to improve productivity by providing an automated AI pair programmer for every programmer in the world. On the cautionary side, since these large language models do not understand program semantics, they offer no guarantees about quality of the suggested code. In this paper, we present an approach to augment these large language models with post-processing steps based on program analysis and synthesis techniques, that understand the syntax and semantics of programs. Further, we show that such techniques can make use of user feedback and improve with usage. We present our experiences from building and evaluating such a tool Jigsaw, targeted at synthesizing code for using Python Pandas API using multi-modal inputs. Our experience suggests that as these large language models evolve for synthesizing code from intent, Jigsaw has an important role to play in improving the accuracy of the systems.
</details>





--------------------------------------------------------------------------------------------------------------------------
#### [ICML'24] Language Agent Tree Search Unifies Reasoning, Acting, and Planning in Language Models
[[paper]](https://arxiv.org/abs/2310.04406)  [[project]](https://github.com/lapisrocks/LanguageAgentTreeSearch) 

<details>
  <summary>Click to see the abstract!</summary>
>  While language models (LMs) have shown potential across a range of decision-making tasks, their reliance on simple acting processes limits their broad deployment as autonomous agents. In this paper, we introduce Language Agent Tree Search (LATS) -- the first general framework that synergizes the capabilities of LMs in reasoning, acting, and planning. By leveraging the in-context learning ability of LMs, we integrate Monte Carlo Tree Search into LATS to enable LMs as agents, along with LM-powered value functions and self-reflections for proficient exploration and enhanced decision-making. A key feature of our approach is the incorporation of an environment for external feedback, which offers a more deliberate and adaptive problem-solving mechanism that surpasses the constraints of existing techniques. Our experimental evaluation across diverse domains, including programming, interactive question-answering (QA), web navigation, and math, validates the effectiveness and generality of LATS in decision-making while maintaining competitive or improved reasoning performance. Notably, LATS achieves state-of-the-art pass@1 accuracy (92.7%) for programming on HumanEval with GPT-4 and demonstrates gradient-free performance (average score of 75.9) comparable to gradient-based fine-tuning for web navigation on WebShop with GPT-3.5. Code can be found at [this https URL](https://github.com/lapisrocks/LanguageAgentTreeSearch)
</details>


# 2. Security in Code Generation: Approaches and Case Studies



--------------------------------------------------------------------------------------------------------------------------
#### [CCS'17] Security Weaknesses of Copilot Generated Code in GitHub

[[paper]](https://arxiv.org/pdf/2310.02059) 

<details>
  <summary>Click to see the abstract!</summary>
> Modern code generation tools, utilizing AI models like Large Language Models (LLMs), have gained popularity for producing functional code. However, their usage presents security challenges, often resulting in insecure code merging into the code base. Evaluating the quality of generated code, especially its security, is crucial. While prior research explored various aspects of code generation, the focus on security has been limited, mostly examining code produced in controlled environments rather than real-world scenarios. To address this gap, we conducted an empirical study, analyzing code snippets generated by GitHub Copilot from GitHub projects. Our analysis identified 452 snippets generated by Copilot, revealing a high likelihood of security issues, with 32.8% of Python and 24.5% of JavaScript snippets affected. These issues span 38 different Common Weakness Enumeration (CWE) categories, including significant ones like CWE-330: Use of Insufficiently Random Values, CWE-78: OS Command Injection, and CWE-94: Improper Control of Generation of Code. Notably, eight CWEs are among the 2023 CWE Top-25, highlighting their severity. Our findings confirm that developers should be careful when adding code generated by Copilot and should also run appropriate security checks as they accept the suggested code. It also shows that practitioners should cultivate corresponding security awareness and skills.
</details>




--------------------------------------------------------------------------------------------------------------------------

#### [arxiv'21]  Evaluating Large Language Models Trained on Code

[[paper]](https://arxiv.org/pdf/2107.03374) [[project]](https://github.com/openai/human-eval)

<details>
  <summary>Click to see the abstract!</summary>
> We introduce Codex, a GPT language model fine-tuned on publicly available code from GitHub, and study its Python code-writing capabilities. A distinct production version of Codex powers GitHub Copilot. On HumanEval, a new evaluation set we release to measure functional correctness for synthesizing programs from docstrings, our model solves 28.8% of the problems, while GPT-3 solves 0% and GPT-J solves 11.4%. Furthermore, we find that repeated sampling from the model is a surprisingly effective strategy for producing working solutions to difficult prompts. Using this method, we solve 70.2% of our problems with 100 samples per problem. Careful investigation of our model reveals its limitations, including difficulty with docstrings describing long chains of operations and with binding operations to variables. Finally, we discuss the potential broader impacts of deploying powerful code generation technologies, covering safety, security, and economics.
</details>





--------------------------------------------------------------------------------------------------------------------------
#### [CCS'23] Do Users Write More Insecure Code with AI Assistants?
[[paper]](https://arxiv.org/pdf/2211.03622)  [[project]](https://github.com/NeilAPerry/Do-Users-Write-More-Insecure-Code-with-AI-Assistants)  

<details>
  <summary>Click to see the abstract!</summary>
>  AI code assistants have emerged as powerful tools that can aid in the software development life-cycle and can improve developer productivity. Unfortunately, such assistants have also been found to produce insecure code in lab environments, raising significant concerns about their usage in practice. In this paper, we conduct a user study to examine how users interact with AI code assistants to solve a variety of security related tasks. Overall, we find that participants who had access to an AI assistant wrote significantly less secure code than those without access to an assistant. Participants with access to an AI assistant were also more likely to believe they wrote secure code, suggesting that such tools may lead users to be overconfident about security flaws in their code. To better inform the design of future AI-based code assistants, we release our user-study apparatus to researchers seeking to build on our work.
</details>



--------------------------------------------------------------------------------------------------------------------------
#### [USENIX Security'23] CodexLeaks: Privacy leaks from code generation language models in GitHub copilot
[[paper]](https://www.usenix.org/system/files/usenixsecurity23-niu.pdf)  [[project]](https://github.com/niuliang42/CodexLeaks/tree/main)  [[talk]](https://youtu.be/sYb84JMl-b4)

<details>
  <summary>Click to see the abstract!</summary>
>  Code generation language models are trained on billions of lines of source code to provide code generation and auto-completion features, like those offered by code assistant GitHub Copilot with more than a million users. These datasets may contain sensitive personal information—personally identifiable, private, or secret—that these models may regurgitate.
>  This paper introduces and evaluates a semi-automated pipeline for extracting sensitive personal information from the Codex model used in GitHub Copilot. We employ carefully-designed templates to construct prompts that are more likely to result in privacy leaks. To overcome the non-public training data, we propose a semi-automated filtering method using a blind membership inference attack. We validate the effectiveness of our membership inference approach on different code generation models. We utilize hit rate through the GitHub Search API as a distinguishing heuristic followed by human-in-the-loop evaluation, uncovering that approximately 8% (43) of the prompts yield privacy leaks. Notably, we observe that the model tends to produce indirect leaks, compromising privacy as contextual integrity by generating information from individuals closely related to the queried subject in the training corpus.
</details>





--------------------------------------------------------------------------------------------------------------------------
#### [CCS'23] Large Language Models for Code: Security Hardening and Adversarial Testing
[[paper]](https://dl.acm.org/doi/pdf/10.1145/3576915.3623175)  [[project]](https://github.com/eth-sri/sven)  
<details>
  <summary>Click to see the abstract!</summary>
> Large language models (large LMs) are increasingly trained on massive codebases and used to generate code. However, LMs lack awareness of security and are found to frequently produce unsafe code. This work studies the security of LMs along two important axes: (i) security hardening, which aims to enhance LMs' reliability in generating secure code, and (ii) adversarial testing, which seeks to evaluate LMs' security at an adversarial standpoint. We address both of these by formulating a new security task called controlled code generation. The task is parametric and takes as input a binary property to guide the LM to generate secure or unsafe code, while preserving the LM's capability of generating functionally correct code. We propose a novel learning-based approach called SVEN to solve this task. SVEN leverages property-specific continuous vectors to guide program generation towards the given property, without modifying the LM's weights. Our training procedure optimizes these continuous vectors by enforcing specialized loss terms on different regions of code, using a high-quality dataset carefully curated by us. Our extensive evaluation shows that SVEN is highly effective in achieving strong security control. For instance, a state-of-the-art CodeGen LM with 2.7B parameters generates secure code for 59.1% of the time. When we employ SVEN to perform security hardening (or adversarial testing) on this LM, the ratio is significantly boosted to 92.3% (or degraded to 36.8%). Importantly, SVEN closely matches the original LMs in functional correctness.
</details>





--------------------------------------------------------------------------------------------------------------------------

#### [usenixsecurity'23]  Lost at C: A User Study on the Security Implications of Large Language Model Code Assistants

[[paper]](https://www.usenix.org/system/files/usenixsecurity23-sandoval.pdf)

<details>
  <summary>Click to see the abstract!</summary>
> Large Language Models (LLMs) such as OpenAI Codex are increasingly being used as AI-based coding assistants. Understanding the impact of these tools on developers’ code is paramount, especially as recent work showed that LLMs may suggest cybersecurity vulnerabilities. We conduct a security-driven user study (N=58) to assess code written by student programmers when assisted by LLMs. Given the potential severity of low-level bugs as well as their relative frequency in real-world projects, we tasked participants with implementing a singly-linked ‘shopping list’ structure in C. Our results indicate that the security impact in this setting (low-level C with pointer and array manipulations) is small: AI-assisted users produce critical security bugs at a rate no greater than 10% more than the control, indicating the use of LLMs does not introduce new security risks.
</details>





--------------------------------------------------------------------------------------------------------------------------

#### [ACM SIGSAC'23]  Large Language Models for Code: Security Hardening and Adversarial Testing

[[paper]](https://dl.acm.org/doi/pdf/10.1145/3576915.3623175)

<details>
  <summary>Click to see the abstract!</summary>
> Large language models (large LMs) are increasingly trained on massive codebases and used to generate code. However, LMs lack awareness of security and are found to frequently produce unsafe code. This work studies the security of LMs along two important axes: (i) security hardening, which aims to enhance LMs' reliability in generating secure code, and (ii) adversarial testing, which seeks to evaluate LMs' security at an adversarial standpoint. We address both of these by formulating a new security task called controlled code generation. The task is parametric and takes as input a binary property to guide the LM to generate secure or unsafe code, while preserving the LM's capability of generating functionally correct code. We propose a novel learning-based approach called SVEN to solve this task. SVEN leverages property-specific continuous vectors to guide program generation towards the given property, without modifying the LM's weights. Our training procedure optimizes these continuous vectors by enforcing specialized loss terms on different regions of code, using a high-quality dataset carefully curated by us. Our extensive evaluation shows that SVEN is highly effective in achieving strong security control. For instance, a state-of-the-art CodeGen LM with 2.7B parameters generates secure code for 59.1% of the time. When we employ SVEN to perform security hardening (or adversarial testing) on this LM, the ratio is significantly boosted to 92.3% (or degraded to 36.8%). Importantly, SVEN closely matches the original LMs in functional correctness.
</details>





--------------------------------------------------------------------------------------------------------------------------

#### [arxiv'24]  Robustness, Security, Privacy, Explainability, Efficiency, and Usability of Large Language Models for Code

[[paper]](https://arxiv.org/pdf/2403.07506)

<details>
  <summary>Click to see the abstract!</summary>
> Large language models for code (LLM4Code), which demonstrate strong performance (e.g., high accuracy) in processing source code, have significantly transformed software engineering. Many studies separately investigate the non-functional properties of LM4Code, but there is no systematic review of how these properties are evaluated and enhanced. This paper fills this gap by thoroughly examining 146 relevant studies, thereby presenting the first systematic literature review to identify seven important properties beyond accuracy, including robustness, security, privacy, explainability, efficiency, and usability. We discuss the current state-of-the-art methods and trends, identify gaps in existing research, and present promising directions for future study.
</details>





--------------------------------------------------------------------------------------------------------------------------

#### [arxiv'24]  Codexity: Secure AI-assisted Code Generation

[[paper]](https://arxiv.org/pdf/2405.03927) [[project]](https://github.com/Codexity-APR/Codexity)

<details>
  <summary>Click to see the abstract!</summary>
> Despite the impressive performance of Large Language Models (LLMs) in software development activities, recent studies show the concern of introducing vulnerabilities into software codebase by AI programming assistants (e.g., Copilot, CodeWhisperer). In this work, we present Codexity, a security-focused code generation framework integrated with five LLMs. Codexity leverages the feedback of static analysis tools such as Infer and CppCheck to mitigate security vulnerabilities in LLM-generated programs. Our evaluation in a real-world benchmark with 751 automatically generated vulnerable subjects demonstrates Codexity can prevent 60% of the vulnerabilities being exposed to the software developer.
</details>





--------------------------------------------------------------------------------------------------------------------------

#### [arxiv'24]  DeVAIC: A Tool for Security Assessment of AI-generated Code

[[paper]](https://arxiv.org/pdf/2404.07548)

<details>
  <summary>Click to see the abstract!</summary>
> Context: AI code generators are revolutionizing code writing and software development, but their training on large datasets, including potentially untrusted source code, raises security concerns. Furthermore, these generators can produce incomplete code snippets that are challenging to evaluate using current solutions. Objective: This research work introduces DeVAIC (Detection of Vulnerabilities in AI-generated Code), a tool to evaluate the security of AI-generated Python code, which overcomes the challenge of examining incomplete code. Method: We followed a methodological approach that involved gathering vulnerable samples, extracting implementation patterns, and creating regular expressions to develop the proposed tool. The implementation of DeVAIC includes a set of detection rules based on regular expressions that cover 35 Common Weakness Enumerations (CWEs) falling under the OWASP Top 10 vulnerability categories. Results: We utilized four popular AI models to generate Python code, which we then used as a foundation to evaluate the effectiveness of our tool. DeVAIC demonstrated a statistically significant difference in its ability to detect security vulnerabilities compared to the state-of-the-art solutions, showing an F1 Score and Accuracy of 94% while maintaining a low computational cost of 0.14 seconds per code snippet, on average. Conclusions: The proposed tool provides a lightweight and efficient solution for vulnerability detection even on incomplete code.
</details>




--------------------------------------------------------------------------------------------------------------------------

#### [arxiv'24]  Constrained Decoding for Secure Code Generation

[[paper]](https://arxiv.org/pdf/2405.00218)

<details>
  <summary>Click to see the abstract!</summary>
> Code Large Language Models (Code LLMs) have been increasingly used by developers to boost productivity, but they often generate vulnerable code. Thus, there is an urgent need to ensure that code generated by Code LLMs is correct and secure. Previous research has primarily focused on generating secure code, overlooking the fact that secure code also needs to be correct. This oversight can lead to a false sense of security. Currently, the community lacks a method to measure actual progress in this area, and we need solutions that address both security and correctness of code generation.
This paper introduces a new benchmark, CodeGuard+, along with two new metrics, secure-pass@k and secure@kpass, to measure Code LLMs' ability to generate both secure and correct code. Using our new evaluation methods, we show that the state-of-the-art defense technique, prefix tuning, may not be as strong as previously believed, since it generates secure code but sacrifices functional correctness. We also demonstrate that different decoding methods significantly affect the security of Code LLMs.
Furthermore, we explore a new defense direction: constrained decoding for secure code generation. We propose new constrained decoding techniques to generate code that satisfies security and correctness constraints simultaneously. Our results reveal that constrained decoding is more effective than prefix tuning to improve the security of Code LLMs, without requiring a specialized training dataset. Moreover, constrained decoding can be used together with prefix tuning to further improve the security of Code LLMs.
</details>





--------------------------------------------------------------------------------------------------------------------------

#### [arxiv'24]  Enhancing Security of AI-Based Code Synthesis with GitHub Copilot via Cheap and Efficient Prompt-Engineering

[[paper]](https://arxiv.org/pdf/2403.12671)

<details>
  <summary>Click to see the abstract!</summary>
> AI assistants for coding are on the rise. However one of the reasons developers and companies avoid harnessing their full potential is the questionable security of the generated code. This paper first reviews the current state-of-the-art and identifies areas for improvement on this issue. Then, we propose a systematic approach based on prompt-altering methods to achieve better code security of (even proprietary black-box) AI-based code generators such as GitHub Copilot, while minimizing the complexity of the application from the user point-of-view, the computational resources, and operational costs. In sum, we propose and evaluate three prompt altering methods: (1) scenario-specific, (2) iterative, and (3) general clause, while we discuss their combination. Contrary to the audit of code security, the latter two of the proposed methods require no expert knowledge from the user. We assess the effectiveness of the proposed methods on the GitHub Copilot using the OpenVPN project in realistic scenarios, and we demonstrate that the proposed methods reduce the number of insecure generated code samples by up to 16\% and increase the number of secure code by up to 8\%. Since our approach does not require access to the internals of the AI models, it can be in general applied to any AI-based code synthesizer, not only GitHub Copilot.
</details>





--------------------------------------------------------------------------------------------------------------------------

#### [arxiv'24]  Just another copy and paste? Comparing the security vulnerabilities of ChatGPT generated code and StackOverflow answers

[[paper]](https://arxiv.org/pdf/2403.15600)

<details>
  <summary>Click to see the abstract!</summary>
> Sonatype's 2023 report found that 97% of developers and security leads integrate generative Artificial Intelligence (AI), particularly Large Language Models (LLMs), into their development process. Concerns about the security implications of this trend have been raised. Developers are now weighing the benefits and risks of LLMs against other relied-upon information sources, such as StackOverflow (SO), requiring empirical data to inform their choice. In this work, our goal is to raise software developers awareness of the security implications when selecting code snippets by empirically comparing the vulnerabilities of ChatGPT and StackOverflow. To achieve this, we used an existing Java dataset from SO with security-related questions and answers. Then, we asked ChatGPT the same SO questions, gathering the generated code for comparison. After curating the dataset, we analyzed the number and types of Common Weakness Enumeration (CWE) vulnerabilities of 108 snippets from each platform using CodeQL. ChatGPT-generated code contained 248 vulnerabilities compared to the 302 vulnerabilities found in SO snippets, producing 20% fewer vulnerabilities with a statistically significant difference. Additionally, ChatGPT generated 19 types of CWE, fewer than the 22 found in SO. Our findings suggest developers are under-educated on insecure code propagation from both platforms, as we found 274 unique vulnerabilities and 25 types of CWE. Any code copied and pasted, created by AI or humans, cannot be trusted blindly, requiring good software engineering practices to reduce risk. Future work can help minimize insecure code propagation from any platform.
</details>





--------------------------------------------------------------------------------------------------------------------------

#### [ICSE-SEIP '24]  PrivacyCAT: Privacy-Aware Code Analysis at Scale

[[paper]](https://dl.acm.org/doi/pdf/10.1145/3639477.3639742)

<details>
  <summary>Click to see the abstract!</summary>
> Static and dynamic code analyses have been widely adopted in industry to enhance software reliability, security, and performance by automatically detecting bugs in the code. In this paper, we introduce PrivacyCAT1, a code analysis system developed and deployed at WhatsApp to protect user privacy. PrivacyCAT automatically detects privacy defects in code at early stages (before reaching production and affecting users), and therefore, it prevents such vulnerabilities from evolving into privacy incidents. PrivacyCAT comprises of a collection of static and dynamic taint analysers.
We report on the technical development of PrivacyCAT and the results of two years of its large-scale industrial deployment at WhatsApp. We present our experience in designing its system architecture, and continuous integration process. We discuss the unique challenges encountered in developing and deploying such kind of analyses within an industrial context.
Since its deployment in 2021, PrivacyCAT has safeguarded data privacy in 74% of privacy site events (SEVs). It has prevented 493 potential privacy SEVs from being introduced into the codebases, enabling developers to maintain a high privacy standard for the code that supports over two billion WhatsApp users.
</details>





--------------------------------------------------------------------------------------------------------------------------

#### [IEEE Transactions on Software Engineering'24]  No Need to Lift a Finger Anymore? Assessing the Quality of Code Generation by ChatGPT

[[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10507163)

<details>
  <summary>Click to see the abstract!</summary>
> Large language models (LLMs) have demonstrated impressive capabilities across various natural language processing (NLP) tasks, such as machine translation, question answering, summarization, and so on. Additionally, LLMs are also highly valuable in supporting software engineering tasks, particularly in the field of code generation. Automatic code generation is a process of automatically generating source code or executable code based on given specifications or requirements, improving developer productivity. In this study, we perform a systematic empirical assessment to the quality of code generation using ChatGPT , a recent state-of-the-art product LLM. We leverage 728 algorithm problems in five languages (i.e., C, C++, Java, Python, and JavaScript) and 18 CWEs with 54 code scenarios for the code generation task. Our evaluation encompasses a comprehensive analysis of code snippets generated by ChatGPT , focusing on three critical aspects: correctness, complexity, and security. We also specifically investigate ChatGPT ’s ability to engage in multi-round fixing process (i.e., ChatGPT ’s dialog ability, chatting between users and ChatGPT for fixing generated buggy code) of facilitating code generation. By delving into the generated code and examining the experimental results, this work provides valuable insights into the performance of ChatGPT in tackling code generation tasks over the three critical aspects. The experimental results demonstrate that (1) ChatGPT is better at generating functionally correct code for problems before 2021 in different languages than problems after 2021 with 48.14% advantage in Accepted rate on judgment platform, but ChatGPT ’s ability to directly fix erroneous code with multi-round fixing process to achieve correct functionality is relatively weak; (2) the distribution of cyclomatic and cognitive complexity levels for code snippets in different languages varies. Furthermore, the multi-round fixing process with ChatGPT generally preserves or increases the complexity levels of code snippets; (3) in algorithm scenarios with languages of C, C++, and Jave, and CWE scenarios with languages of C and Python3, the code generated by ChatGPT has relevant vulnerabilities. However, the multi-round fixing process for vulnerable code snippets demonstrates promising results, with more than 89% of vulnerabilities successfully addressed; and (4) code generation may be affected by ChatGPT ’s non-determinism factor, resulting in variations of code snippets in functional correctness, complexity, and security. Overall, our findings uncover potential issues and limitations that arise in the ChatGPT -based code generation and lay the groundwork for improving AI and LLM-based code generation techniques.
</details>




--------------------------------------------------------------------------------------------------------------------------

#### [IEEE SaTML'24]  CodeLMSec Benchmark: Systematically Evaluating and Finding Security Vulnerabilities in Black-Box Code Language Models

[[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10516658)

<details>
  <summary>Click to see the abstract!</summary>
> Large language models (LLMs) for automatic code generation have recently achieved breakthroughs in several programming tasks. Their advances in competition-level programming problems have made them an essential pillar of AI-assisted pair programming, and tools such as GitHub Copilot have emerged as part of the daily programming workflow used by millions of developers. Training data for these models is usually collected from the Internet (e.g., from open-source repositories) and is likely to contain faults and security vulnerabilities. This unsanitized training data can cause the language models to learn these vulnerabilities and propagate them during the code generation procedure. While these models have been extensively evaluated for their ability to produce functionally correct programs, there remains a lack of comprehensive investigations and benchmarks addressing the security aspects of these models.In this work, we propose a method to systematically study the security issues of code language models to assess their susceptibility to generating vulnerable code. To this end, we introduce the first approach to automatically find generated code that contains vulnerabilities in black-box code generation models. This involves proposing a novel few-shot prompting approach. We evaluate the effectiveness of our approach by examining code language models in generating high-risk security weaknesses. Furthermore, we use our method to create a collection of diverse non-secure prompts for various vulnerability scenarios. This dataset serves as a benchmark to evaluate and compare the security weaknesses of code language models.
</details>





--------------------------------------------------------------------------------------------------------------------------

#### [ACM SE'24]  A Pilot Study on Secure Code Generation with ChatGPT for Web Applications

[[paper]](https://dl.acm.org/doi/abs/10.1145/3603287.3651194)

<details>
  <summary>Click to see the abstract!</summary>
> Conversational Large Language Models (LLMs), such as ChatGPT, have demonstrated their potent capabilities in natural language processing tasks. This paper presents a pilot study that uses ChatGPT for generating web application code with a specific emphasis on mitigating four prevalent web application vulnerability types: SQL Injection, Cross Site Scripting, Carriage Return Line Feed Injection, and Exposure of Sensitive Information. The paper uses a case study to illustrate how the vulnerabilities in the code are mitigated with the prompts and the subsequent refinements. The study's findings summarize the security concerns in the code generated by ChatGPT, and the paper proposes a prompt pattern designed to help mitigating the potential vulnerabilities.
</details>



--------------------------------------------------------------------------------------------------------------------------

#### [Journal of Systems and Software'24]  Automating the correctness assessment of AI-generated code for security contexts

[[paper]](https://www.sciencedirect.com/science/article/pii/S0164121224001584)

<details>
  <summary>Click to see the abstract!</summary>
> Evaluating the correctness of code generated by AI is a challenging open problem. In this paper, we propose a fully automated method, named ACCA, to evaluate the correctness of AI-generated code for security purposes. The method uses symbolic execution to assess whether the AI-generated code behaves as a reference implementation. We use ACCA to assess four state-of-the-art models trained to generate security-oriented assembly code and compare the results of the evaluation with different baseline solutions, including output similarity metrics, widely used in the field, and the well-known ChatGPT, the AI-powered language model developed by OpenAI.
Our experiments show that our method outperforms the baseline solutions and assesses the correctness of the AI-generated code similar to the human-based evaluation, which is considered the ground truth for the assessment in the field. Moreover, ACCA has a very strong correlation with the human evaluation (Pearson’s correlation coefficient r=0.84 on average). Finally, since it is a full y automated solution that does not require any human intervention, the proposed method performs the assessment of every code snippet in 0.17 s on average, which is definitely lower than the average time required by human analysts to manually inspect the code, based on our experience.
</details>



--------------------------------------------------------------------------------------------------------------------------

#### [Frontiers in Big Data'24]  A systematic literature review on the impact of AI models on the security of code generation

[[paper]](https://www.frontiersin.org/articles/10.3389/fdata.2024.1386720/full)

<details>
  <summary>Click to see the abstract!</summary>
> Artificial Intelligence (AI) is increasingly used as a helper to develop computing programs. While it can boost software development and improve coding proficiency, this practice offers no guarantee of security. On the contrary, recent research shows that some AI models produce software with vulnerabilities. This situation leads to the question: How serious and widespread are the security flaws in code generated using AI models?
</details>



--------------------------------------------------------------------------------------------------------------------------

#### [High-Confidence Computing'24]  A Survey on Large Language Model (LLM) Security and Privacy: The Good, The Bad, and The Ugly

[[paper]](https://www.sciencedirect.com/science/article/pii/S266729522400014X)

<details>
  <summary>Click to see the abstract!</summary>
> Large Language Models (LLMs), such as ChatGPT and Bard, have revolutionized natural language understanding and generation. They possess deep language comprehension, human-like text generation capabilities, contextual awareness, and robust problem-solving skills, making them invaluable in various domains (e.g., search engines, customer support, translation). In the meantime, LLMs have also gained traction in the security community, revealing security vulnerabilities and showcasing their potential in security-related tasks. This paper explores the intersection of LLMs with security and privacy. Specifically, we investigate how LLMs positively impact security and privacy, potential risks and threats associated with their use, and inherent vulnerabilities within LLMs. Through a comprehensive literature review, the paper categorizes the papers into “The Good” (beneficial LLM applications), “The Bad” (offensive applications), and “The Ugly” (vulnerabilities of LLMs and their defenses). We have some interesting findings. For example, LLMs have proven to enhance code security (code vulnerability detection) and data privacy (data confidentiality protection), outperforming traditional methods. However, they can also be harnessed for various attacks (particularly user-level attacks) due to their human-like reasoning abilities. We have identified areas that require further research efforts. For example, Research on model and parameter extraction attacks is limited and often theoretical, hindered by LLM parameter scale and confidentiality. Safe instruction tuning, a recent development, requires more exploration. We hope that our work can shed light on the LLMs’ potential to both bolster and jeopardize cybersecurity.
</details>







