#!/bin/bash

OUTPUT_DIR="data/raw"
mkdir -p "$OUTPUT_DIR"

SUCCESS=0
SKIP=0
FAIL=0

download() {
  local ID="$1"
  local NAME="$2"
  local OUTFILE="$OUTPUT_DIR/${NAME}.pdf"

  if [ -f "$OUTFILE" ]; then
    echo "SKIP (already exists): $NAME"
    ((SKIP++))
    return
  fi

  echo "Downloading: $NAME..."
  HTTP_CODE=$(curl -L --fail --silent --show-error \
    -w "%{http_code}" \
    "https://arxiv.org/pdf/${ID}" \
    -o "$OUTFILE")

  if [ "$HTTP_CODE" -eq 200 ]; then
    echo "  OK: $NAME"
    ((SUCCESS++))
  else
    echo "  FAIL: $NAME (HTTP $HTTP_CODE)"
    rm -f "$OUTFILE"
    ((FAIL++))
  fi

  sleep 1
}

# Foundation Architecture
download 1706.03762 "Attention_Is_All_You_Need"
download 1810.04805 "BERT"
download 2005.14165 "GPT-3"
download 2307.09288 "LLaMA_2"
download 2310.06825 "Mistral_7B"
download 2404.14219 "Phi-3_Technical_Report"
download 2407.10671 "Qwen2_Technical_Report"
download 2401.02385 "TinyLlama"

# RAG & Retrieval
download 2005.11401 "RAG_Lewis_et_al"
download 2310.11511 "Self-RAG"
download 2401.18059 "RAPTOR"
download 2212.10496 "HyDE"
download 2004.12832 "ColBERT"
download 2307.03172 "Lost_in_the_Middle"
download 1908.10084 "Sentence-BERT"
download 2309.07597 "BGE_Embedding"
download 2401.15884 "Corrective_RAG"

# Efficient Inference & Quantization
download 2106.09685 "LoRA"
download 2305.14314 "QLoRA"
download 2210.17323 "GPTQ"
download 2306.00978 "AWQ"
download 2208.07339 "LLM_int8"
download 2205.14135 "FlashAttention"
download 2307.08691 "FlashAttention_2"
download 2211.17192 "Speculative_Decoding"

# Agents & Reasoning
download 2201.11903 "Chain-of-Thought_Prompting"
download 2210.03629 "ReAct"
download 2305.10601 "Tree_of_Thoughts"
download 2302.04761 "Toolformer"
download 2205.00445 "MRKL_LangChain_Agent"
download 2203.11171 "Self-Consistency"
download 2303.11366 "Reflexion"

# Alignment & Safety
download 2203.02155 "InstructGPT_RLHF"
download 2212.08073 "Constitutional_AI"
download 2305.18290 "DPO"
download 2202.03629 "Hallucination_Survey"
download 2109.07958 "TruthfulQA"

# Vision & Multimodal
download 2010.11929 "ViT_Vision_Transformer"
download 2103.00020 "CLIP"
download 2304.07193 "DINOv2"
download 2304.02643 "SAM_Segment_Anything"
download 2304.08485 "LLaVA"
download 2303.08774 "GPT-4V_Technical_Report"
download 2204.14198 "Flamingo"

# Architecture Details
download 1502.03167 "Batch_Normalization"
download 1607.06450 "Layer_Normalization"
download 2104.09864 "RoPE_Rotary_Position_Embedding"
download 2305.13245 "GQA_Grouped_Query_Attention"
download 2101.03961 "Mixture_of_Experts"
download 1512.03385 "ResNet"

# Foundation LLMs (additional)
download 2204.02311 "PaLM"
download 2305.10403 "PaLM_2"
download 2403.05530 "Gemini_1.5"
download 2407.21783 "LLaMA_3"
download 2405.04434 "DeepSeek_V2"
download 2403.08295 "Gemma"
download 2205.01068 "OPT"
download 2211.05100 "BLOOM"
download 1910.10683 "T5"
download 1906.08237 "XLNet"
download 1907.11692 "RoBERTa"
download 2109.01652 "FLAN"
download 2203.15556 "Chinchilla_Scaling"
download 2001.08361 "Scaling_Laws_for_LLMs"
download 2206.07682 "Emergent_Abilities_of_LLMs"
download 2305.13048 "RWKV"
download 2312.00752 "Mamba"
download 1909.11942 "ALBERT"
download 2003.10555 "ELECTRA"
download 2501.12948 "DeepSeek_R1"

# Efficient Inference & Training
download 2004.05150 "Longformer"
download 2007.14062 "BigBird"
download 2001.04451 "Reformer"
download 2307.08621 "RetNet"
download 2309.06180 "vLLM_PagedAttention"
download 2211.10438 "SmoothQuant"
download 1711.05101 "AdamW"
download 1710.03740 "Mixed_Precision_Training"
download 1910.02054 "ZeRO_DeepSpeed"
download 1503.02531 "Knowledge_Distillation"

# Diffusion & Generative Models
download 2006.11239 "DDPM"
download 2010.02502 "DDIM"
download 2112.10752 "Latent_Diffusion_Stable_Diffusion"
download 2102.12092 "DALL-E"
download 2205.11487 "Imagen"
download 2011.13456 "Score_Based_Generative_Models"
download 2210.02747 "Flow_Matching"
download 2212.09748 "DiT_Diffusion_Transformer"
download 2302.05543 "ControlNet"
download 1406.2661  "GAN"
download 1912.04958 "StyleGAN2"
download 1312.6114  "VAE"
download 1906.00446 "VQ-VAE-2"
download 2204.06125 "DALL-E_2"
download 2304.08818 "VideoLDM"

# Computer Vision
download 2201.03545 "ConvNeXt"
download 2103.14030 "Swin_Transformer"
download 2012.12877 "DeiT"
download 2111.06377 "MAE_Masked_Autoencoder"
download 2106.08254 "BEiT"
download 2104.14294 "DINO_Self_Supervised_Vision"
download 2002.05709 "SimCLR"
download 2006.07733 "BYOL"
download 1905.11946 "EfficientNet"
download 2005.12872 "DETR"
download 1703.06870 "Mask_R-CNN"
download 1506.01497 "Faster_R-CNN"
download 2004.10934 "YOLOv4"
download 2003.13678 "RegNet"
download 2201.09792 "ConvMixer"

# Multimodal
download 2201.12086 "BLIP"
download 2301.12597 "BLIP-2"
download 2305.06500 "InstructBLIP"
download 2304.10592 "MiniGPT-4"
download 2305.05665 "ImageBind"
download 2308.12966 "Qwen-VL"
download 2311.03079 "CogVLM"
download 2312.14238 "InternVL"
download 2306.02858 "Video-LLaMA"
download 2212.04356 "Whisper"

# RAG & Retrieval (additional)
download 2004.04906 "DPR_Dense_Passage_Retrieval"
download 2002.08909 "REALM"
download 2007.01282 "FiD_Fusion_in_Decoder"
download 2104.08663 "BEIR_Benchmark"
download 2107.05720 "SPLADE"
download 2112.09118 "Contriever"
download 2212.03533 "E5_Embedding"
download 2210.07316 "MTEB_Benchmark"
download 2112.04426 "RETRO"
download 2203.05115 "Internet_Augmented_LLM"

# Agents & Tools
download 2303.17580 "HuggingGPT"
download 2308.08155 "AutoGen"
download 2308.00352 "MetaGPT"
download 2307.16789 "ToolBench"
download 2305.15334 "Gorilla_API_Calling"
download 2308.03688 "AgentBench"
download 2402.01030 "CodeAct"
download 2310.10634 "OpenAgents"
download 2303.16434 "TaskMatrix"
download 2405.15793 "SWE-agent"

# Alignment & Safety (additional)
download 1707.06347 "PPO"
download 2402.03300 "GRPO"
download 2305.11206 "LIMA"
download 2212.10560 "Self-Instruct"
download 2304.07327 "OpenAssistant"
download 2202.03286 "Red_Teaming_LLMs"
download 2403.13787 "RewardBench"
download 2307.01232 "RLHF_Survey"
download 2402.01306 "KTO"
download 2310.12773 "Safe_RLHF"

# Benchmarks & Evaluation
download 2009.03300 "MMLU"
download 2107.03374 "HumanEval_Codex"
download 2110.14168 "GSM8K"
download 2206.04615 "BIG-Bench"
download 2211.09110 "HELM"
download 2306.05685 "MT-Bench"
download 2310.06770 "SWE-bench"
download 2103.03874 "MATH_Benchmark"
download 2304.06364 "AGIEval"
download 2403.04132 "Chatbot_Arena"

# Code & Programming
download 2305.06161 "StarCoder"
download 2308.12950 "CodeLlama"
download 2401.14196 "DeepSeek_Coder"
download 2203.07814 "AlphaCode"
download 2306.08568 "WizardCoder"

# Speech & Audio
download 2209.03143 "AudioLM"
download 2301.11325 "MusicLM"
download 2210.13438 "EnCodec"
download 2006.11477 "Wav2Vec_2"
download 2306.15687 "Voicebox"

# Graph Neural Networks
download 1609.02907 "GCN"
download 1710.10903 "GAT"
download 1706.02216 "GraphSAGE"
download 1902.10197 "RotatE"
download 2106.05234 "Graphormer"

# Domain Applications
download 2212.13138 "Med-PaLM"
download 2210.10341 "BioGPT"
download 1903.10676 "SciBERT"
download 2202.07622 "ESMFold"
download 2211.09085 "Galactica"
download 2303.14070 "ChatDoctor"
download 2204.11817 "MolT5"
download 2306.06031 "FinGPT"
download 2308.11462 "LegalBench"
download 2303.17564 "BloombergGPT"

# Reasoning & Math
download 2211.12588 "Program_of_Thoughts"
download 2211.10435 "PAL"
download 2205.10625 "Least_to_Most_Prompting"
download 2310.06117 "Step_Back_Prompting"
download 2112.00114 "Scratchpad_Show_Your_Work"
download 2310.03731 "MathCoder"
download 2308.09583 "WizardMath"
download 2305.20050 "Process_Reward_Model"
download 2403.09629 "Quiet-STaR"
download 2203.14465 "STaR_Self_Taught_Reasoner"

# Misc Foundational
download 1301.3666  "Word2Vec"
download 1607.04606 "FastText"
download 1808.05377 "NAS_Survey"
download 2110.04544 "CLIP_Adapter"
download 2104.08691 "Prompt_Tuning"

# VLA & Embodied AI
download 2212.06817 "RT-1_Robotics_Transformer"
download 2307.15818 "RT-2_VLA"
download 2310.08864 "Open_X-Embodiment"
download 2406.09246 "OpenVLA"
download 2209.11171 "Code_as_Policies"
download 2204.01691 "SayCan"
download 2303.04137 "Diffusion_Policy"
download 2305.16291 "Voyager"
download 2311.01378 "RoboFlamingo"
download 2309.10190 "EmbodiedGPT"
download 2401.17539 "ManipLLM"

# 2024-2025 Foundation LLMs
download 2401.04088 "Mixtral_8x7B"
download 2412.19437 "DeepSeek_V3"
download 2408.01313 "Gemma_2"
download 2402.00838 "OLMo"
download 2310.16944 "Zephyr"
download 2306.11644 "Phi_1"
download 2309.05463 "Phi_1.5"
download 2403.04652 "Yi_34B"
download 2403.19887 "Jamba"
download 2402.07827 "Aya_101"
download 2405.16338 "Phi_3.5_Mini"
download 2403.17297 "InternLM_2"

# Multimodal 2024-2025
download 2310.03744 "LLaVA_1.5"
download 2404.16821 "InternVL_1.5"
download 2409.12191 "Qwen2_VL"
download 2405.02246 "Idefics2"
download 2312.13286 "Emu2"
download 2306.16527 "IDEFICS"
download 2408.00714 "SAM_2"
download 2412.05271 "InternVL_2.5"
download 2404.02882 "MiniCPM_V"
download 2406.16860 "LLaVA_NeXT_Video"
download 2406.07476 "VideoLLaMA2"
download 2405.09841 "LLaVA_NeXT_Interleave"

# Long Context & Memory
download 2309.12307 "LongLoRA"
download 2310.06839 "Ring_Attention"
download 2308.14508 "LongBench"
download 2310.07600 "MemGPT"
download 2402.14848 "Infini_Attention"
download 2309.00071 "YaRN"

# Efficient Fine-tuning 2024
download 2402.09353 "DoRA"
download 2403.03507 "GaLore"
download 2307.13269 "LoRAHub"
download 2309.14717 "LoftQ"

# 3D Vision & NeRF
download 2003.08934 "NeRF"
download 2103.13415 "Mip_NeRF"
download 2201.05989 "Instant_NGP"
download 2308.04079 "3D_Gaussian_Splatting"
download 2304.06706 "Zip_NeRF"
download 2309.07920 "GaussianAvatars"

# Video Generation
download 2209.14430 "Make-A-Video"
download 2307.04725 "AnimateDiff"
download 2311.17062 "AnimateDiff_V2"
download 2309.13038 "Show-1"
download 2406.05358 "CogVideoX"
download 2312.14730 "VideoPoet"

# RAG & Retrieval 2024
download 2404.16130 "GraphRAG"
download 2312.10997 "RAG_Survey_2024"
download 2405.14831 "HippoRAG"
download 2410.05779 "LightRAG"
download 2403.14403 "Adaptive_RAG"
download 2407.16526 "LongRAG"
download 2404.06283 "RAFT"
download 2310.11511 "FLARE"

# Agentic AI 2024
download 2309.03409 "OPRO"
download 2303.17760 "CAMEL"
download 2307.07924 "ChatDev"
download 2308.10848 "AgentVerse"
download 2406.12753 "AGENTLESS"
download 2402.01817 "OSWorld"

# Safety & Alignment 2024
download 2312.06674 "Llama_Guard"
download 2310.01405 "Representation_Engineering"
download 2309.00267 "RLAIF"
download 2401.01335 "SPIN_Self_Play"
download 2406.04040 "Llama_Guard_2"
download 2404.14054 "WildGuard"

# Benchmarks 2024
download 2310.02255 "MathVista"
download 2311.07911 "IFEval"
download 2406.04770 "MMLU_Pro"
download 2406.19314 "LiveBench"
download 2307.13854 "WebArena"
download 2404.14022 "WildBench"
download 2311.12022 "HaluEval"

# Code Generation 2024
download 2406.11931 "DeepSeek_Coder_V2"
download 2402.19173 "StarCoder2"
download 2312.02120 "Magicoder"
download 2407.01461 "CRUXEval"

# On-Device & Edge AI
download 2402.14905 "MobileLLM"
download 2312.11382 "On_Device_LLM_Survey"
download 2406.02691 "EdgeShard"
download 2405.14672 "LLM_Edge_Survey"
download 2407.10019 "MobileVLM_V2"

# Optimization & Training
download 1412.6980  "Adam_Optimizer"
download 2302.06675 "Lion_Optimizer"
download 2305.14342 "Sophia_Optimizer"
download 1904.00962 "LAMB_Optimizer"
download 2006.16668 "GShard_MoE"
download 2202.09368 "Expert_Choice_MoE"

# NLP Architecture Classics
download 1910.13461 "BART"
download 2006.03654 "DeBERTa"
download 2010.11934 "mT5"
download 2108.12409 "ALiBi"
download 1910.07467 "RMSNorm"
download 1905.03197 "UniLM"

# Multi-Agent
download 2305.14325 "LLM_Society_of_Mind"
download 2308.09830 "AgentSims"
download 2211.01910 "APE_Auto_Prompt_Engineer"
download 2309.16797 "Promptbreeder"

# Continual & Federated Learning
download 1612.00796 "EWC_Continual_Learning"
download 1602.05629 "FedAvg"
download 1812.06127 "FedProx"

# Retrieval & Embedding 2024
download 2401.00368 "E5_Mistral_Instruct"
download 2402.15449 "NV_Embed"
download 2405.04434 "BGE_M3"
download 2404.12699 "Gecko_Embedding"

# Reasoning 2024
download 2411.14405 "Marco_o1"
download 2401.10020 "Evol_Instruct_Reasoning"
download 2312.11805 "SPIN_Reasoning"
download 2406.12793 "Skywork_o1"

# Vision Foundation 2024
download 2303.05499 "Grounding_DINO"
download 2205.06230 "OWL-ViT"
download 2112.03857 "GLIP"
download 2405.14458 "Grounding_DINO_1.5"
download 2404.07177 "DINO_v2_Extended"

echo ""
echo "Done.  Success: $SUCCESS / Skip: $SKIP / Fail: $FAIL  (Total target: 500 papers)"
