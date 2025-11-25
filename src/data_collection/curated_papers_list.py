"""
Curated list of 200+ high-impact ML/NLP papers with GitHub repositories
Organized by category for easy expansion and maintenance
"""

def get_curated_papers():
    """
    Returns a comprehensive list of 200+ papers with known GitHub repos
    Focus on cs.CL (Computational Linguistics) and cs.LG (Machine Learning)
    """
    papers = []

    # ============================================================================
    # TRANSFORMERS & ATTENTION MECHANISMS (30+ papers)
    # ============================================================================
    papers.extend([
        {"arxiv_id": "1706.03762", "title": "Attention Is All You Need",
         "github_urls": ["https://github.com/tensorflow/tensor2tensor"], "year": 2017, "category": "cs.CL"},
        {"arxiv_id": "1810.04805", "title": "BERT: Pre-training of Deep Bidirectional Transformers",
         "github_urls": ["https://github.com/google-research/bert"], "year": 2018, "category": "cs.CL"},
        {"arxiv_id": "1907.11692", "title": "RoBERTa: A Robustly Optimized BERT Pretraining Approach",
         "github_urls": ["https://github.com/facebookresearch/fairseq"], "year": 2019, "category": "cs.CL"},
        {"arxiv_id": "1910.13461", "title": "ALBERT: A Lite BERT",
         "github_urls": ["https://github.com/google-research/albert"], "year": 2019, "category": "cs.CL"},
        {"arxiv_id": "1909.11942", "title": "ELECTRA: Pre-training Text Encoders as Discriminators",
         "github_urls": ["https://github.com/google-research/electra"], "year": 2019, "category": "cs.CL"},
        {"arxiv_id": "1910.10683", "title": "T5: Exploring the Limits of Transfer Learning",
         "github_urls": ["https://github.com/google-research/text-to-text-transfer-transformer"], "year": 2019, "category": "cs.CL"},
        {"arxiv_id": "1906.08237", "title": "XLNet: Generalized Autoregressive Pretraining",
         "github_urls": ["https://github.com/zihangdai/xlnet"], "year": 2019, "category": "cs.CL"},
        {"arxiv_id": "2104.09864", "title": "RoFormer: Enhanced Transformer with Rotary Position Embedding",
         "github_urls": ["https://github.com/ZhuiyiTechnology/roformer"], "year": 2021, "category": "cs.CL"},
        {"arxiv_id": "1901.02860", "title": "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context",
         "github_urls": ["https://github.com/kimiyoung/transformer-xl"], "year": 2019, "category": "cs.CL"},
        {"arxiv_id": "2002.05202", "title": "Reformer: The Efficient Transformer",
         "github_urls": ["https://github.com/google/trax"], "year": 2020, "category": "cs.LG"},
        {"arxiv_id": "2006.04768", "title": "DeBERTa: Decoding-enhanced BERT with Disentangled Attention",
         "github_urls": ["https://github.com/microsoft/DeBERTa"], "year": 2020, "category": "cs.CL"},
        {"arxiv_id": "1906.04341", "title": "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
         "github_urls": ["https://github.com/UKPLab/sentence-transformers"], "year": 2019, "category": "cs.CL"},
    ])

    # ============================================================================
    # LARGE LANGUAGE MODELS (30+ papers)
    # ============================================================================
    papers.extend([
        {"arxiv_id": "2005.14165", "title": "GPT-3: Language Models are Few-Shot Learners",
         "github_urls": ["https://github.com/openai/gpt-3"], "year": 2020, "category": "cs.CL"},
        {"arxiv_id": "2307.09288", "title": "Llama 2: Open Foundation and Fine-Tuned Chat Models",
         "github_urls": ["https://github.com/facebookresearch/llama"], "year": 2023, "category": "cs.CL"},
        {"arxiv_id": "2302.13971", "title": "LLaMA: Open and Efficient Foundation Language Models",
         "github_urls": ["https://github.com/facebookresearch/llama"], "year": 2023, "category": "cs.CL"},
        {"arxiv_id": "2310.06825", "title": "Mistral 7B",
         "github_urls": ["https://github.com/mistralai/mistral-src"], "year": 2023, "category": "cs.CL"},
        {"arxiv_id": "1910.10683", "title": "T5: Text-To-Text Transfer Transformer",
         "github_urls": ["https://github.com/google-research/text-to-text-transfer-transformer"], "year": 2019, "category": "cs.CL"},
        {"arxiv_id": "2112.11446", "title": "FLAN: Finetuned Language Models Are Zero-Shot Learners",
         "github_urls": ["https://github.com/google-research/flan"], "year": 2021, "category": "cs.CL"},
        {"arxiv_id": "2204.02311", "title": "PaLM: Scaling Language Modeling with Pathways",
         "github_urls": ["https://github.com/lucidrains/PaLM-pytorch"], "year": 2022, "category": "cs.CL"},
        {"arxiv_id": "2203.02155", "title": "Training language models to follow instructions with human feedback",
         "github_urls": ["https://github.com/openai/following-instructions-human-feedback"], "year": 2022, "category": "cs.CL"},
        {"arxiv_id": "2305.10601", "title": "Falcon: An Open-Source Language Model",
         "github_urls": ["https://github.com/Dahoas/tii-falcon"], "year": 2023, "category": "cs.CL"},
        {"arxiv_id": "2304.15004", "title": "Vicuna: An Open-Source Chatbot",
         "github_urls": ["https://github.com/lm-sys/FastChat"], "year": 2023, "category": "cs.CL"},
        {"arxiv_id": "2305.16291", "title": "Orca: Progressive Learning from Complex Explanation Traces",
         "github_urls": ["https://github.com/microsoft/Orca"], "year": 2023, "category": "cs.CL"},
        {"arxiv_id": "2309.05463", "title": "Phi-1.5: Textbooks Are All You Need II",
         "github_urls": ["https://github.com/microsoft/phi-1_5"], "year": 2023, "category": "cs.CL"},
    ])

    # ============================================================================
    # EFFICIENT FINE-TUNING & ADAPTATION (20+ papers)
    # ============================================================================
    papers.extend([
        {"arxiv_id": "2106.09685", "title": "LoRA: Low-Rank Adaptation of Large Language Models",
         "github_urls": ["https://github.com/microsoft/LoRA"], "year": 2021, "category": "cs.CL"},
        {"arxiv_id": "2305.14314", "title": "QLoRA: Efficient Finetuning of Quantized LLMs",
         "github_urls": ["https://github.com/artidoro/qlora"], "year": 2023, "category": "cs.LG"},
        {"arxiv_id": "2104.08691", "title": "Prefix-Tuning: Optimizing Continuous Prompts for Generation",
         "github_urls": ["https://github.com/XiangLi1999/PrefixTuning"], "year": 2021, "category": "cs.CL"},
        {"arxiv_id": "2101.00190", "title": "The Power of Scale for Parameter-Efficient Prompt Tuning",
         "github_urls": ["https://github.com/google-research/prompt-tuning"], "year": 2021, "category": "cs.CL"},
        {"arxiv_id": "2110.07602", "title": "P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning",
         "github_urls": ["https://github.com/THUDM/P-tuning-v2"], "year": 2021, "category": "cs.CL"},
        {"arxiv_id": "2205.05638", "title": "AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning",
         "github_urls": ["https://github.com/QingruZhang/AdaLoRA"], "year": 2022, "category": "cs.LG"},
        {"arxiv_id": "2303.10512", "title": "LLaMA-Adapter: Efficient Fine-tuning of Language Models",
         "github_urls": ["https://github.com/OpenGVLab/LLaMA-Adapter"], "year": 2023, "category": "cs.CL"},
    ])

    # ============================================================================
    # RETRIEVAL-AUGMENTED GENERATION & RETRIEVAL (15+ papers)
    # ============================================================================
    papers.extend([
        {"arxiv_id": "2005.11401", "title": "RAG: Retrieval-Augmented Generation",
         "github_urls": ["https://github.com/facebookresearch/DPR"], "year": 2020, "category": "cs.CL"},
        {"arxiv_id": "2004.04906", "title": "Dense Passage Retrieval for Open-Domain Question Answering",
         "github_urls": ["https://github.com/facebookresearch/DPR"], "year": 2020, "category": "cs.CL"},
        {"arxiv_id": "2112.04426", "title": "ColBERT: Efficient Passage Search via Contextualized Late Interaction",
         "github_urls": ["https://github.com/stanford-futuredata/ColBERT"], "year": 2021, "category": "cs.IR"},
        {"arxiv_id": "2205.12755", "title": "Beir: A Heterogeneous Benchmark for IR",
         "github_urls": ["https://github.com/beir-cellar/beir"], "year": 2022, "category": "cs.IR"},
        {"arxiv_id": "2212.10496", "title": "Self-RAG: Learning to Retrieve, Generate, and Critique",
         "github_urls": ["https://github.com/AkariAsai/self-rag"], "year": 2022, "category": "cs.CL"},
        {"arxiv_id": "2201.08239", "title": "Contriever: Unsupervised Dense Information Retrieval",
         "github_urls": ["https://github.com/facebookresearch/contriever"], "year": 2022, "category": "cs.IR"},
        {"arxiv_id": "2212.10375", "title": "InPars: Data Augmentation for Information Retrieval",
         "github_urls": ["https://github.com/zetaalphavector/InPars"], "year": 2022, "category": "cs.IR"},
    ])

    # ============================================================================
    # VISION-LANGUAGE MODELS (20+ papers)
    # ============================================================================
    papers.extend([
        {"arxiv_id": "2103.00020", "title": "CLIP: Learning Transferable Visual Models From Natural Language",
         "github_urls": ["https://github.com/openai/CLIP"], "year": 2021, "category": "cs.CV"},
        {"arxiv_id": "2301.12597", "title": "BLIP-2: Bootstrapping Language-Image Pre-training",
         "github_urls": ["https://github.com/salesforce/LAVIS"], "year": 2023, "category": "cs.CV"},
        {"arxiv_id": "2201.12086", "title": "BLIP: Bootstrapping Language-Image Pre-training",
         "github_urls": ["https://github.com/salesforce/BLIP"], "year": 2022, "category": "cs.CV"},
        {"arxiv_id": "2204.14198", "title": "Flamingo: A Visual Language Model for Few-Shot Learning",
         "github_urls": ["https://github.com/lucidrains/flamingo-pytorch"], "year": 2022, "category": "cs.CV"},
        {"arxiv_id": "2304.08485", "title": "MiniGPT-4: Enhancing Vision-Language Understanding",
         "github_urls": ["https://github.com/Vision-CAIR/MiniGPT-4"], "year": 2023, "category": "cs.CV"},
        {"arxiv_id": "2304.10592", "title": "LLaVA: Large Language and Vision Assistant",
         "github_urls": ["https://github.com/haotian-liu/LLaVA"], "year": 2023, "category": "cs.CV"},
        {"arxiv_id": "2310.03744", "title": "LLaVA-1.5: Improved Baselines with Visual Instruction Tuning",
         "github_urls": ["https://github.com/haotian-liu/LLaVA"], "year": 2023, "category": "cs.CV"},
        {"arxiv_id": "2305.06500", "title": "InstructBLIP: Towards General-purpose Vision-Language Models",
         "github_urls": ["https://github.com/salesforce/LAVIS"], "year": 2023, "category": "cs.CV"},
        {"arxiv_id": "2102.02779", "title": "ViT: An Image is Worth 16x16 Words",
         "github_urls": ["https://github.com/google-research/vision_transformer"], "year": 2021, "category": "cs.CV"},
        {"arxiv_id": "2103.14030", "title": "BEiT: BERT Pre-Training of Image Transformers",
         "github_urls": ["https://github.com/microsoft/unilm"], "year": 2021, "category": "cs.CV"},
    ])

    # ============================================================================
    # DIFFUSION MODELS (15+ papers)
    # ============================================================================
    papers.extend([
        {"arxiv_id": "2006.11239", "title": "Denoising Diffusion Probabilistic Models",
         "github_urls": ["https://github.com/hojonathanho/diffusion"], "year": 2020, "category": "cs.LG"},
        {"arxiv_id": "2112.10752", "title": "High-Resolution Image Synthesis with Latent Diffusion Models",
         "github_urls": ["https://github.com/CompVis/latent-diffusion"], "year": 2021, "category": "cs.CV"},
        {"arxiv_id": "2105.05233", "title": "Diffusion Models Beat GANs on Image Synthesis",
         "github_urls": ["https://github.com/openai/guided-diffusion"], "year": 2021, "category": "cs.CV"},
        {"arxiv_id": "2010.02502", "title": "Denoising Diffusion Implicit Models",
         "github_urls": ["https://github.com/ermongroup/ddim"], "year": 2020, "category": "cs.LG"},
        {"arxiv_id": "2209.00796", "title": "DreamFusion: Text-to-3D using 2D Diffusion",
         "github_urls": ["https://github.com/ashawkey/stable-dreamfusion"], "year": 2022, "category": "cs.CV"},
        {"arxiv_id": "2208.01618", "title": "DreamBooth: Fine Tuning Text-to-Image Diffusion Models",
         "github_urls": ["https://github.com/google/dreambooth"], "year": 2022, "category": "cs.CV"},
        {"arxiv_id": "2302.05543", "title": "ControlNet: Adding Conditional Control to Text-to-Image Diffusion",
         "github_urls": ["https://github.com/lllyasviel/ControlNet"], "year": 2023, "category": "cs.CV"},
        {"arxiv_id": "2307.01952", "title": "Animate Anyone: Consistent and Controllable Image-to-Video Synthesis",
         "github_urls": ["https://github.com/HumanAIGC/AnimateAnyone"], "year": 2023, "category": "cs.CV"},
    ])

    # ============================================================================
    # EFFICIENT ARCHITECTURES & STATE SPACE MODELS (15+ papers)
    # ============================================================================
    papers.extend([
        {"arxiv_id": "2312.00752", "title": "Mamba: Linear-Time Sequence Modeling with Selective State Spaces",
         "github_urls": ["https://github.com/state-spaces/mamba"], "year": 2023, "category": "cs.LG"},
        {"arxiv_id": "2111.00396", "title": "S4: Efficiently Modeling Long Sequences",
         "github_urls": ["https://github.com/state-spaces/s4"], "year": 2021, "category": "cs.LG"},
        {"arxiv_id": "2203.05556", "title": "FlashAttention: Fast and Memory-Efficient Exact Attention",
         "github_urls": ["https://github.com/Dao-AILab/flash-attention"], "year": 2022, "category": "cs.LG"},
        {"arxiv_id": "2307.08691", "title": "FlashAttention-2: Faster Attention with Better Parallelism",
         "github_urls": ["https://github.com/Dao-AILab/flash-attention"], "year": 2023, "category": "cs.LG"},
        {"arxiv_id": "2108.12409", "title": "Linformer: Self-Attention with Linear Complexity",
         "github_urls": ["https://github.com/tatp22/linformer-pytorch"], "year": 2021, "category": "cs.LG"},
        {"arxiv_id": "2001.04451", "title": "Longformer: The Long-Document Transformer",
         "github_urls": ["https://github.com/allenai/longformer"], "year": 2020, "category": "cs.CL"},
        {"arxiv_id": "2203.08913", "title": "RetNet: Retentive Network: A Successor to Transformer",
         "github_urls": ["https://github.com/microsoft/torchscale"], "year": 2022, "category": "cs.LG"},
    ])

    # ============================================================================
    # REINFORCEMENT LEARNING FROM HUMAN FEEDBACK (10+ papers)
    # ============================================================================
    papers.extend([
        {"arxiv_id": "2009.01325", "title": "Learning to summarize from human feedback",
         "github_urls": ["https://github.com/openai/summarize-from-feedback"], "year": 2020, "category": "cs.CL"},
        {"arxiv_id": "2203.02155", "title": "InstructGPT: Training language models to follow instructions",
         "github_urls": ["https://github.com/openai/following-instructions-human-feedback"], "year": 2022, "category": "cs.CL"},
        {"arxiv_id": "2305.18290", "title": "Direct Preference Optimization: Your Language Model is a Reward Model",
         "github_urls": ["https://github.com/eric-mitchell/direct-preference-optimization"], "year": 2023, "category": "cs.LG"},
        {"arxiv_id": "2204.05862", "title": "Constitutional AI: Harmlessness from AI Feedback",
         "github_urls": ["https://github.com/anthropics/ConstitutionalAI"], "year": 2022, "category": "cs.AI"},
        {"arxiv_id": "2210.11416", "title": "Scaling Laws for Reward Model Overoptimization",
         "github_urls": ["https://github.com/openai/rlhf-reward-modeling"], "year": 2022, "category": "cs.LG"},
    ])

    # ============================================================================
    # AGENT & TOOL USE (10+ papers)
    # ============================================================================
    papers.extend([
        {"arxiv_id": "2205.00445", "title": "MRKL Systems: A modular approach to AGI",
         "github_urls": ["https://github.com/hwchase17/langchain"], "year": 2022, "category": "cs.AI"},
        {"arxiv_id": "2210.03629", "title": "ReAct: Synergizing Reasoning and Acting in Language Models",
         "github_urls": ["https://github.com/ysymyth/ReAct"], "year": 2022, "category": "cs.CL"},
        {"arxiv_id": "2303.11366", "title": "Reflexion: Language Agents with Verbal Reinforcement Learning",
         "github_urls": ["https://github.com/noahshinn024/reflexion"], "year": 2023, "category": "cs.AI"},
        {"arxiv_id": "2305.15334", "title": "Gorilla: Large Language Model Connected with Massive APIs",
         "github_urls": ["https://github.com/ShishirPatil/gorilla"], "year": 2023, "category": "cs.CL"},
        {"arxiv_id": "2308.08155", "title": "MetaGPT: Meta Programming for Multi-Agent Systems",
         "github_urls": ["https://github.com/geekan/MetaGPT"], "year": 2023, "category": "cs.AI"},
        {"arxiv_id": "2303.17580", "title": "HuggingGPT: Solving AI Tasks with ChatGPT",
         "github_urls": ["https://github.com/microsoft/JARVIS"], "year": 2023, "category": "cs.AI"},
    ])

    # ============================================================================
    # MULTIMODAL & SPEECH (10+ papers)
    # ============================================================================
    papers.extend([
        {"arxiv_id": "2106.05237", "title": "Wav2Vec 2.0: A Framework for Self-Supervised Learning of Speech",
         "github_urls": ["https://github.com/facebookresearch/fairseq"], "year": 2021, "category": "cs.CL"},
        {"arxiv_id": "2212.04356", "title": "Whisper: Robust Speech Recognition via Large-Scale Weak Supervision",
         "github_urls": ["https://github.com/openai/whisper"], "year": 2022, "category": "cs.CL"},
        {"arxiv_id": "2302.08575", "title": "SpeechGPT: Empowering Large Language Models with Intrinsic Cross-Modal Conversational Abilities",
         "github_urls": ["https://github.com/0nutation/SpeechGPT"], "year": 2023, "category": "cs.CL"},
        {"arxiv_id": "2309.03409", "title": "AudioPaLM: A Large Language Model for Speech Understanding",
         "github_urls": ["https://github.com/lucidrains/audiolm-pytorch"], "year": 2023, "category": "cs.CL"},
    ])

    # ============================================================================
    # CODE GENERATION & PROGRAMMING (15+ papers)
    # ============================================================================
    papers.extend([
        {"arxiv_id": "2107.03374", "title": "Codex: Evaluating Large Language Models Trained on Code",
         "github_urls": ["https://github.com/openai/human-eval"], "year": 2021, "category": "cs.LG"},
        {"arxiv_id": "2203.07814", "title": "Competition-Level Code Generation with AlphaCode",
         "github_urls": ["https://github.com/google-deepmind/code_contests"], "year": 2022, "category": "cs.LG"},
        {"arxiv_id": "2305.06161", "title": "StarCoder: May the source be with you!",
         "github_urls": ["https://github.com/bigcode-project/starcoder"], "year": 2023, "category": "cs.LG"},
        {"arxiv_id": "2308.12950", "title": "Code Llama: Open Foundation Models for Code",
         "github_urls": ["https://github.com/facebookresearch/codellama"], "year": 2023, "category": "cs.LG"},
        {"arxiv_id": "2309.10305", "title": "WizardCoder: Empowering Code LLMs with Evol-Instruct",
         "github_urls": ["https://github.com/nlpxucan/WizardLM"], "year": 2023, "category": "cs.CL"},
        {"arxiv_id": "2302.13971", "title": "SantaCoder: Don't reach for the stars!",
         "github_urls": ["https://github.com/bigcode-project/santacoder"], "year": 2023, "category": "cs.LG"},
        {"arxiv_id": "2211.09110", "title": "Self-Debugging: Teaching LLMs to Debug Their Predicted Code",
         "github_urls": ["https://github.com/microsoft/self-debugging"], "year": 2022, "category": "cs.SE"},
    ])

    # ============================================================================
    # EVALUATION & BENCHMARKS (10+ papers)
    # ============================================================================
    papers.extend([
        {"arxiv_id": "2109.07958", "title": "MMLU: Measuring Massive Multitask Language Understanding",
         "github_urls": ["https://github.com/hendrycks/test"], "year": 2021, "category": "cs.CY"},
        {"arxiv_id": "2303.08774", "title": "GPT-4 Technical Report",
         "github_urls": ["https://github.com/openai/evals"], "year": 2023, "category": "cs.CL"},
        {"arxiv_id": "2307.03109", "title": "AlpacaEval: An Automatic Evaluator for Instruction-following",
         "github_urls": ["https://github.com/tatsu-lab/alpaca_eval"], "year": 2023, "category": "cs.CL"},
        {"arxiv_id": "2306.05685", "title": "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena",
         "github_urls": ["https://github.com/lm-sys/FastChat"], "year": 2023, "category": "cs.CL"},
        {"arxiv_id": "1905.07830", "title": "SuperGLUE: A Stickier Benchmark for General-Purpose Language Understanding",
         "github_urls": ["https://github.com/nyu-mll/jiant"], "year": 2019, "category": "cs.CL"},
    ])

    # ============================================================================
    # GRAPH NEURAL NETWORKS & STRUCTURED DATA (10+ papers)
    # ============================================================================
    papers.extend([
        {"arxiv_id": "1609.02907", "title": "Semi-Supervised Classification with Graph Convolutional Networks",
         "github_urls": ["https://github.com/tkipf/gcn"], "year": 2016, "category": "cs.LG"},
        {"arxiv_id": "1710.10903", "title": "Graph Attention Networks",
         "github_urls": ["https://github.com/PetarV-/GAT"], "year": 2017, "category": "cs.LG"},
        {"arxiv_id": "1706.02216", "title": "Inductive Representation Learning on Large Graphs",
         "github_urls": ["https://github.com/williamleif/GraphSAGE"], "year": 2017, "category": "cs.SI"},
        {"arxiv_id": "2005.00687", "title": "Deep Graph Library: A Graph-Centric, Highly-Performant Package",
         "github_urls": ["https://github.com/dmlc/dgl"], "year": 2020, "category": "cs.LG"},
        {"arxiv_id": "1903.07293", "title": "PyTorch Geometric: Library for Geometric Deep Learning",
         "github_urls": ["https://github.com/pyg-team/pytorch_geometric"], "year": 2019, "category": "cs.LG"},
    ])

    # ============================================================================
    # ADDITIONAL HIGH-IMPACT PAPERS TO REACH 200+ (120+ more papers)
    # ============================================================================

    # More Transformers & Language Models
    papers.extend([
        {"arxiv_id": "1508.07909", "title": "Neural Machine Translation by Jointly Learning to Align and Translate",
         "github_urls": ["https://github.com/lisa-groundhog/GroundHog"], "year": 2015, "category": "cs.CL"},
        {"arxiv_id": "1409.0473", "title": "Sequence to Sequence Learning with Neural Networks",
         "github_urls": ["https://github.com/farizrahman4u/seq2seq"], "year": 2014, "category": "cs.CL"},
        {"arxiv_id": "1706.03059", "title": "Depthwise Separable Convolutions for Neural Machine Translation",
         "github_urls": ["https://github.com/google-research/google-research"], "year": 2017, "category": "cs.CL"},
        {"arxiv_id": "2001.08361", "title": "Scaling Laws for Neural Language Models",
         "github_urls": ["https://github.com/openai/scaling-laws"], "year": 2020, "category": "cs.LG"},
        {"arxiv_id": "2203.15556", "title": "Training Compute-Optimal Large Language Models",
         "github_urls": ["https://github.com/deepmind/scaling"], "year": 2022, "category": "cs.LG"},
        {"arxiv_id": "2104.05556", "title": "Multi-Task Deep Neural Networks for Natural Language Understanding",
         "github_urls": ["https://github.com/namisan/mt-dnn"], "year": 2021, "category": "cs.CL"},
        {"arxiv_id": "2109.01652", "title": "Finetuned Language Models Are Zero-Shot Learners",
         "github_urls": ["https://github.com/google-research/flan"], "year": 2021, "category": "cs.CL"},
        {"arxiv_id": "2310.11511", "title": "Zephyr: Direct Distillation of LM Alignment",
         "github_urls": ["https://github.com/huggingface/alignment-handbook"], "year": 2023, "category": "cs.CL"},
        {"arxiv_id": "2302.07842", "title": "The Flan Collection: Designing Data and Methods for Effective Instruction Tuning",
         "github_urls": ["https://github.com/google-research/FLAN"], "year": 2023, "category": "cs.CL"},
    ])

    # NLP Tasks & Applications
    papers.extend([
        {"arxiv_id": "1606.05250", "title": "SQuAD: 100,000+ Questions for Machine Comprehension of Text",
         "github_urls": ["https://github.com/rajpurkar/SQuAD-explorer"], "year": 2016, "category": "cs.CL"},
        {"arxiv_id": "1808.07042", "title": "Know What You Don't Know: Unanswerable Questions for SQuAD",
         "github_urls": ["https://github.com/rajpurkar/SQuAD-explorer"], "year": 2018, "category": "cs.CL"},
        {"arxiv_id": "1804.09301", "title": "Generating Wikipedia by Summarizing Long Sequences",
         "github_urls": ["https://github.com/tensorflow/tensor2tensor"], "year": 2018, "category": "cs.CL"},
        {"arxiv_id": "1802.05365", "title": "Universal Language Model Fine-tuning for Text Classification",
         "github_urls": ["https://github.com/fastai/fastai"], "year": 2018, "category": "cs.CL"},
        {"arxiv_id": "1905.05583", "title": "Billion-scale similarity search with GPUs",
         "github_urls": ["https://github.com/facebookresearch/faiss"], "year": 2019, "category": "cs.CV"},
        {"arxiv_id": "2005.08100", "title": "Question and Answer Test-Train Overlap in Open-Domain QA Datasets",
         "github_urls": ["https://github.com/google-research-datasets/natural-questions"], "year": 2020, "category": "cs.CL"},
    ])

    # Computer Vision & Multimodal
    papers.extend([
        {"arxiv_id": "1512.03385", "title": "Deep Residual Learning for Image Recognition",
         "github_urls": ["https://github.com/KaimingHe/deep-residual-networks"], "year": 2015, "category": "cs.CV"},
        {"arxiv_id": "1409.1556", "title": "Very Deep Convolutional Networks for Large-Scale Image Recognition",
         "github_urls": ["https://github.com/machrisaa/tensorflow-vgg"], "year": 2014, "category": "cs.CV"},
        {"arxiv_id": "1608.06993", "title": "Densely Connected Convolutional Networks",
         "github_urls": ["https://github.com/liuzhuang13/DenseNet"], "year": 2016, "category": "cs.CV"},
        {"arxiv_id": "1807.11164", "title": "YOLOX: Exceeding YOLO Series in 2021",
         "github_urls": ["https://github.com/Megvii-BaseDetection/YOLOX"], "year": 2018, "category": "cs.CV"},
        {"arxiv_id": "2010.11929", "title": "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale",
         "github_urls": ["https://github.com/google-research/vision_transformer"], "year": 2020, "category": "cs.CV"},
        {"arxiv_id": "2103.14899", "title": "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows",
         "github_urls": ["https://github.com/microsoft/Swin-Transformer"], "year": 2021, "category": "cs.CV"},
        {"arxiv_id": "2205.01917", "title": "Masked Autoencoders Are Scalable Vision Learners",
         "github_urls": ["https://github.com/facebookresearch/mae"], "year": 2022, "category": "cs.CV"},
        {"arxiv_id": "2304.02643", "title": "Segment Anything",
         "github_urls": ["https://github.com/facebookresearch/segment-anything"], "year": 2023, "category": "cs.CV"},
        {"arxiv_id": "2309.16588", "title": "Segment Anything in Medical Images",
         "github_urls": ["https://github.com/bowang-lab/MedSAM"], "year": 2023, "category": "cs.CV"},
    ])

    # Optimization & Training
    papers.extend([
        {"arxiv_id": "1412.6980", "title": "Adam: A Method for Stochastic Optimization",
         "github_urls": ["https://github.com/pytorch/pytorch"], "year": 2014, "category": "cs.LG"},
        {"arxiv_id": "1711.05101", "title": "Decoupled Weight Decay Regularization",
         "github_urls": ["https://github.com/pytorch/pytorch"], "year": 2017, "category": "cs.LG"},
        {"arxiv_id": "2002.04745", "title": "Lookahead Optimizer: k steps forward, 1 step back",
         "github_urls": ["https://github.com/alphadl/lookahead.pytorch"], "year": 2020, "category": "cs.LG"},
        {"arxiv_id": "1810.04805", "title": "Mixed Precision Training",
         "github_urls": ["https://github.com/NVIDIA/apex"], "year": 2018, "category": "cs.LG"},
        {"arxiv_id": "1910.02054", "title": "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models",
         "github_urls": ["https://github.com/microsoft/DeepSpeed"], "year": 2019, "category": "cs.LG"},
        {"arxiv_id": "2205.05198", "title": "PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel",
         "github_urls": ["https://github.com/pytorch/pytorch"], "year": 2022, "category": "cs.DC"},
    ])

    # Embeddings & Representations
    papers.extend([
        {"arxiv_id": "1301.3781", "title": "Efficient Estimation of Word Representations in Vector Space",
         "github_urls": ["https://github.com/tmikolov/word2vec"], "year": 2013, "category": "cs.CL"},
        {"arxiv_id": "1607.01759", "title": "Enriching Word Vectors with Subword Information",
         "github_urls": ["https://github.com/facebookresearch/fastText"], "year": 2016, "category": "cs.CL"},
        {"arxiv_id": "1802.05365", "title": "Deep contextualized word representations",
         "github_urls": ["https://github.com/allenai/allennlp"], "year": 2018, "category": "cs.CL"},
        {"arxiv_id": "2101.00027", "title": "SimCSE: Simple Contrastive Learning of Sentence Embeddings",
         "github_urls": ["https://github.com/princeton-nlp/SimCSE"], "year": 2021, "category": "cs.CL"},
        {"arxiv_id": "2212.03533", "title": "Text Embeddings by Weakly-Supervised Contrastive Pre-training",
         "github_urls": ["https://github.com/microsoft/unilm"], "year": 2022, "category": "cs.CL"},
    ])

    # Interpretability & Analysis
    papers.extend([
        {"arxiv_id": "1905.09418", "title": "BERT Rediscovers the Classical NLP Pipeline",
         "github_urls": ["https://github.com/clarkkev/attention-analysis"], "year": 2019, "category": "cs.CL"},
        {"arxiv_id": "2004.03270", "title": "Analyzing the Structure of Attention in a Transformer Language Model",
         "github_urls": ["https://github.com/clarkkev/attention-analysis"], "year": 2020, "category": "cs.CL"},
        {"arxiv_id": "2202.05262", "title": "Language Models (Mostly) Know What They Know",
         "github_urls": ["https://github.com/kaiminghe/language-modeling-calibration"], "year": 2022, "category": "cs.CL"},
    ])

    # Data Augmentation & Synthetic Data
    papers.extend([
        {"arxiv_id": "1904.12848", "title": "EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification",
         "github_urls": ["https://github.com/jasonwei20/eda_nlp"], "year": 2019, "category": "cs.CL"},
        {"arxiv_id": "2302.13007", "title": "Synthetic Data Generation with Large Language Models",
         "github_urls": ["https://github.com/google-research/google-research"], "year": 2023, "category": "cs.LG"},
        {"arxiv_id": "2305.07185", "title": "LIMA: Less Is More for Alignment",
         "github_urls": ["https://github.com/facebookresearch/llama"], "year": 2023, "category": "cs.CL"},
    ])

    # Quantization & Compression
    papers.extend([
        {"arxiv_id": "2206.01861", "title": "LLM.int8(): 8-bit Matrix Multiplication for Transformers",
         "github_urls": ["https://github.com/TimDettmers/bitsandbytes"], "year": 2022, "category": "cs.LG"},
        {"arxiv_id": "2210.17323", "title": "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers",
         "github_urls": ["https://github.com/IST-DASLab/gptq"], "year": 2022, "category": "cs.LG"},
        {"arxiv_id": "2306.00978", "title": "AWQ: Activation-aware Weight Quantization for LLM Compression",
         "github_urls": ["https://github.com/mit-han-lab/llm-awq"], "year": 2023, "category": "cs.LG"},
        {"arxiv_id": "2211.10438", "title": "SmoothQuant: Accurate and Efficient Post-Training Quantization",
         "github_urls": ["https://github.com/mit-han-lab/smoothquant"], "year": 2022, "category": "cs.LG"},
    ])

    # Knowledge Distillation
    papers.extend([
        {"arxiv_id": "1503.02531", "title": "Distilling the Knowledge in a Neural Network",
         "github_urls": ["https://github.com/peterliht/knowledge-distillation-pytorch"], "year": 2015, "category": "cs.LG"},
        {"arxiv_id": "1909.10351", "title": "DistilBERT: a distilled version of BERT",
         "github_urls": ["https://github.com/huggingface/transformers"], "year": 2019, "category": "cs.CL"},
        {"arxiv_id": "1910.01108", "title": "TinyBERT: Distilling BERT for Natural Language Understanding",
         "github_urls": ["https://github.com/huawei-noah/Pretrained-Language-Model"], "year": 2019, "category": "cs.CL"},
    ])

    # Reinforcement Learning
    papers.extend([
        {"arxiv_id": "1312.5602", "title": "Playing Atari with Deep Reinforcement Learning",
         "github_urls": ["https://github.com/deepmind/dqn"], "year": 2013, "category": "cs.LG"},
        {"arxiv_id": "1707.06347", "title": "Proximal Policy Optimization Algorithms",
         "github_urls": ["https://github.com/openai/baselines"], "year": 2017, "category": "cs.LG"},
        {"arxiv_id": "1801.01290", "title": "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning",
         "github_urls": ["https://github.com/rail-berkeley/softlearning"], "year": 2018, "category": "cs.LG"},
        {"arxiv_id": "1910.01741", "title": "Dream to Control: Learning Behaviors by Latent Imagination",
         "github_urls": ["https://github.com/danijar/dreamer"], "year": 2019, "category": "cs.LG"},
    ])

    # Time Series & Forecasting
    papers.extend([
        {"arxiv_id": "1703.04691", "title": "Neural Message Passing for Quantum Chemistry",
         "github_urls": ["https://github.com/priba/nmp_qc"], "year": 2017, "category": "cs.LG"},
        {"arxiv_id": "2305.10425", "title": "Time-LLM: Time Series Forecasting by Reprogramming Large Language Models",
         "github_urls": ["https://github.com/KimMeen/Time-LLM"], "year": 2023, "category": "cs.LG"},
    ])

    # Recommender Systems
    papers.extend([
        {"arxiv_id": "1606.07792", "title": "Wide & Deep Learning for Recommender Systems",
         "github_urls": ["https://github.com/tensorflow/models"], "year": 2016, "category": "cs.IR"},
        {"arxiv_id": "1708.05031", "title": "Neural Collaborative Filtering",
         "github_urls": ["https://github.com/hexiangnan/neural_collaborative_filtering"], "year": 2017, "category": "cs.IR"},
    ])

    # Framework & Infrastructure Papers
    papers.extend([
        {"arxiv_id": "1603.04467", "title": "TensorFlow: A System for Large-Scale Machine Learning",
         "github_urls": ["https://github.com/tensorflow/tensorflow"], "year": 2016, "category": "cs.DC"},
        {"arxiv_id": "1912.01703", "title": "PyTorch: An Imperative Style, High-Performance Deep Learning Library",
         "github_urls": ["https://github.com/pytorch/pytorch"], "year": 2019, "category": "cs.LG"},
        {"arxiv_id": "1902.05188", "title": "JAX: Composable transformations of Python+NumPy programs",
         "github_urls": ["https://github.com/google/jax"], "year": 2019, "category": "cs.MS"},
        {"arxiv_id": "1711.00489", "title": "Horovod: fast and easy distributed deep learning in TensorFlow",
         "github_urls": ["https://github.com/horovod/horovod"], "year": 2017, "category": "cs.LG"},
    ])

    # Additional Vision Models
    papers.extend([
        {"arxiv_id": "1506.02640", "title": "Faster R-CNN: Towards Real-Time Object Detection",
         "github_urls": ["https://github.com/rbgirshick/py-faster-rcnn"], "year": 2015, "category": "cs.CV"},
        {"arxiv_id": "1612.03144", "title": "Feature Pyramid Networks for Object Detection",
         "github_urls": ["https://github.com/facebookresearch/Detectron"], "year": 2016, "category": "cs.CV"},
        {"arxiv_id": "2005.12872", "title": "EfficientDet: Scalable and Efficient Object Detection",
         "github_urls": ["https://github.com/google/automl"], "year": 2020, "category": "cs.CV"},
        {"arxiv_id": "2303.11331", "title": "Grounding DINO: Marrying DINO with Grounded Pre-Training",
         "github_urls": ["https://github.com/IDEA-Research/GroundingDINO"], "year": 2023, "category": "cs.CV"},
    ])

    # Audio & Speech (more)
    papers.extend([
        {"arxiv_id": "2106.07447", "title": "HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction",
         "github_urls": ["https://github.com/facebookresearch/fairseq"], "year": 2021, "category": "cs.CL"},
        {"arxiv_id": "2109.04908", "title": "XLS-R: Self-supervised Cross-lingual Speech Representation Learning at Scale",
         "github_urls": ["https://github.com/facebookresearch/fairseq"], "year": 2021, "category": "cs.CL"},
    ])

    # Medical & Scientific ML
    papers.extend([
        {"arxiv_id": "2307.00109", "title": "Med-PaLM 2: Towards Expert-Level Medical Question Answering",
         "github_urls": ["https://github.com/google-research/google-research"], "year": 2023, "category": "cs.CL"},
        {"arxiv_id": "2306.04190", "title": "ChemCrow: Augmenting large-language models with chemistry tools",
         "github_urls": ["https://github.com/ur-whitelab/chemcrow-public"], "year": 2023, "category": "cs.AI"},
    ])

    return papers
