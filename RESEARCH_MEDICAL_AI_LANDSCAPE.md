# Medical AI Landscape Research: Voice-Based Emergency Triage System
## Compiled April 2026 — Focus: Kenya MVP, Speed/Latency, Medical Accuracy

---

## 1. Medical LLMs (Open-Source)

### Tier 1: Best Available Open-Source Medical Models

| Model | Size | Base | License | MedQA | PubMedQA | Notes |
|-------|------|------|---------|-------|----------|-------|
| **MedGemma 1.5** (Google, Jan 2026) | 4B multimodal, 27B text/multimodal | Gemma 2 | Open (research + commercial) | Strong (exact score TBD, competitive with Med-PaLM) | Strong | **TOP PICK** — newest, multimodal (X-rays, derm, ophtho), FHIR EHR support in 27B. 4B variant is small enough for fast inference |
| **Llama-3-Meditron 70B** (EPFL, 2024) | 8B, 70B | Llama 3.1 | Llama license (permissive) | ~60%+ (8B outperforms older 70B models) | Near Med-PaLM-2 | Best performing open-source suite. 70B only slightly behind GPT-4-base |
| **OpenBioLLM-70B** (Saama AI, Apr 2024) | 8B, 70B | Llama 3 | Apache 2.0 | Competitive | Strong | Claims to outperform GPT-4 on some medical benchmarks |
| **BioMistral-7B** (2024) | 7B | Mistral 7B | Apache 2.0 | ~52% (4-option) | ~74% | Best in class for 7B size. Pre-trained on PubMed Central. Fast inference |
| **Meditron-70B** (EPFL, 2023) | 7B, 70B | Llama 2 | Llama 2 license | 52% (7B), higher for 70B | 74.4% (7B) | Outperforms GPT-3.5, within 5% of GPT-4 |
| **Me-LLaMA** (2024) | 13B, 70B | Llama 2 | Research | Outperforms PMC-LLaMA, MedAlpaca | Good | Solid mid-tier option |

### Tier 2: Older / More Limited

| Model | Size | Notes |
|-------|------|-------|
| **MedAlpaca** | 7B, 13B | Instruction fine-tuned but limited by base model knowledge |
| **PMC-LLaMA** | 7B, 13B | Continual pretraining on medical literature. Superseded by Me-LLaMA |
| **Clinical Camel** | 70B | LLaMA-2 based, dialogue-focused. Less widely adopted |

### Conversational / Real-Time Suitability

For a voice triage system, the model must support **streaming token generation** and respond within ~1-2 seconds:

- **MedGemma 4B**: Best balance of medical knowledge + speed. Small enough for Groq/fast inference
- **BioMistral-7B**: Fastest option with decent medical knowledge. Runs well quantized
- **Llama-3-Meditron 8B**: Strong medical performance at a conversational-friendly size
- **MedGemma 27B / Meditron 70B**: Too large for real-time voice unless using Groq or high-end GPU cluster

---

## 2. General-Purpose Models on Medical Benchmarks

| Model | MedQA Score | USMLE Avg | Notes |
|-------|-------------|-----------|-------|
| **GPT-5** (OpenAI, late 2025) | **95.8%** | **95.2%** | State of the art. Expensive. API only |
| **Med-Gemini** (Google) | **91.1%** | ~90%+ | Uses uncertainty-guided search. API only |
| **GPT-4o** | ~86% | ~86% | Strong general medical reasoning |
| **Claude 3.5 Sonnet** | ~82-85% (estimated) | ~83-85% | Strong reasoning, good at following clinical protocols |
| **Gemini 1.5 Pro** | ~85% | ~85% | Good multimodal capabilities |
| **GPT-4** (original) | 71.6% (zero-shot) | ~80%+ | Baseline reference |

### Key Insight for MVP

General-purpose models (GPT-4o, Claude, Gemini) **significantly outperform** specialized open-source models on medical benchmarks. The gap is 20-40% on MedQA. For a triage system where accuracy is life-critical:

**Recommendation**: Use a general-purpose model API (GPT-4o-mini or Claude 3.5 Haiku for speed, GPT-4o or Claude Sonnet for accuracy) as primary, with a specialized open-source model as fallback/offline mode.

---

## 3. Fast Inference Options

### Cloud API Providers (Ranked by Latency)

| Provider | Speed | Models Available | TTFT | Throughput | Best For |
|----------|-------|-----------------|------|------------|----------|
| **Groq (LPU)** | **Fastest** | Llama 3.x, Mixtral, Gemma | <100ms TTFT, <300ms consistent | 750-900 tok/s on 70B | **TOP PICK for voice** — 20x faster than GPU, sub-100ms first token |
| **Cerebras** | Very fast | Llama 3.x | <100ms | 500-800 tok/s | Alternative speed leader |
| **Fireworks AI** | Fast | 200+ open models | ~200ms | High | Best for structured output (JSON triage forms). FireAttention = 4x faster than vLLM for JSON |
| **Together AI** | Fast | 200+ open models | <100ms (claimed) | Good | Good balance of speed/cost/model selection |
| **DeepInfra** | Good | Wide selection | ~200ms | Good | Cost-effective for background tasks |

### Self-Hosted Options

| Option | Latency | Hardware Needed | Notes |
|--------|---------|-----------------|-------|
| **vLLM** | 200-500ms TTFT | GPU (A100/H100) | Industry standard. PagedAttention for efficient batching |
| **llama.cpp (GGUF Q4_K_M)** | 300-800ms TTFT (CPU), 100-300ms (GPU) | CPU or consumer GPU | Best for offline/edge. 7B model runs on 8GB RAM |
| **TensorRT-LLM** | 100-200ms | NVIDIA GPU | Fastest self-hosted but complex setup |

### MVP Recommendation

**Primary**: Groq API serving Llama-3-Meditron 8B or MedGemma 4B (if available on Groq)
- Fallback to GPT-4o-mini via OpenAI API for complex cases
- Groq gives you sub-300ms LLM response for voice pipeline

**Offline fallback**: llama.cpp with BioMistral-7B GGUF Q4_K_M on local hardware

---

## 4. Voice AI Stack

### Speech-to-Text (STT)

| Provider | Latency | Swahili Support | WER (English) | WER (Swahili) | Streaming | Price |
|----------|---------|-----------------|---------------|----------------|-----------|-------|
| **Deepgram Nova-3** | **150-184ms** first words | **Limited** (36 languages total) | ~8-10% | Unknown/unlikely | Yes | $0.0043/min |
| **Whisper Large v3** (OpenAI) | 500-2000ms (batch) | **Yes** (99 languages) | ~5-8% | **10-15% WER** | Via streaming wrappers | Free (self-host) or API |
| **AssemblyAI Universal-2** | ~300ms | Limited | ~8% | Unknown | Yes | $0.01/min |
| **Google Cloud STT** | ~200ms | **Yes** (Swahili supported) | ~7% | ~12-15% | Yes | $0.006/min |

**Swahili STT Recommendation**:
- **Primary**: Whisper Large v3 — best Swahili support (10-15% WER), free self-hosted
- Use **faster-whisper** (CTranslate2 backend) for 4x speedup, streaming chunks
- **Fallback**: Google Cloud STT for streaming with Swahili
- Recent research (2025) demonstrated edge-cloud cascaded STT for Swahili using Whisper, tested with Kenyan survey data — bandwidth requirement: ~1 MB/s

### Text-to-Speech (TTS)

| Provider | Latency (TTFA) | Swahili | Quality | Streaming | Price |
|----------|-----------------|---------|---------|-----------|-------|
| **Cartesia Sonic 3** | **40ms (Turbo), 90ms** | 15 languages (check Swahili) | Excellent, emotional | Yes | API pricing |
| **ElevenLabs Flash v2.5** | **75ms** | 29+ languages, Swahili likely | Best naturalness | Yes | $0.18/1K chars |
| **Deepgram Aura-2** | **90ms** | Limited | Good | Yes | $0.015/1K chars |
| **PlayHT** | ~300ms | Multiple languages | Good | Yes | API pricing |
| **Coqui XTTS v2** (open source) | <200ms | 17 languages (Swahili uncertain) | Good, voice cloning from 10s | Yes | **Free** (self-host) |
| **SpeechT5** (open source) | Variable | Swahili (with fine-tuning) | Moderate | Limited | **Free** |

**Swahili TTS Recommendation**:
- **Primary**: ElevenLabs Flash v2.5 — likely Swahili support, 75ms latency, most natural
- **Budget/Offline**: Coqui XTTS v2 self-hosted — can clone a Swahili-speaking voice from 10-second sample
- Recent 2025 research used SpeechT5 for Swahili TTS in a Kenyan edge-cloud framework

### Real-Time Voice Pipeline Frameworks

| Framework | Maintained By | Transport | Key Strength | License |
|-----------|---------------|-----------|--------------|---------|
| **Pipecat** | Daily.co | WebRTC (Daily), WebSocket | **TOP PICK** — vendor-agnostic, used by NVIDIA/Cresta. Plug-and-play STT/LLM/TTS pipeline | Open source (BSD) |
| **LiveKit Agents** | LiveKit | WebRTC | Best turn detection (sub-75ms P99 semantic detector). Production-grade infrastructure | Open source (Apache 2.0) |
| **Vocode** | Vocode | WebSocket, telephony | Good telephony integration (Twilio, Vonage) | Open source |
| **Bolna** | Bolna | WebSocket | Simple setup for phone-based agents | Open source |

**Voice Pipeline MVP Architecture**:
```
User Phone/App
    ↓ (WebRTC or WebSocket)
Pipecat or LiveKit Agent
    ↓
faster-whisper (STT, ~200ms) → Groq/LLM (~200ms) → ElevenLabs Flash (TTS, ~75ms)
    ↓
Audio back to user
    
Total target: 500-800ms end-to-end (conversational)
```

---

## 5. Kenya-Specific Considerations

### Swahili Language Support Summary

| Component | Best Option | Swahili Quality | Offline Capable? |
|-----------|-------------|-----------------|------------------|
| STT | Whisper v3 (via faster-whisper) | 10-15% WER | Yes (self-hosted) |
| TTS | ElevenLabs or Coqui XTTS | Good (EL), trainable (Coqui) | Coqui only |
| LLM | GPT-4o / Claude (multilingual) | Strong Swahili understanding | No (API) |

### Internet Connectivity

- Kenyan ISPs generally offer >1 MB/s download speeds (2024 data)
- This exceeds the minimum for cascaded voice AI (~512 KB/s TTS, ~1 MB/s STT)
- **However**: Rural/remote areas may have unreliable connectivity
- **Recommendation**: Design for graceful degradation — edge STT (Whisper) + cached medical protocols + queue-and-sync for LLM calls

### Common Emergency Scenarios to Build For

Based on Kenyan health statistics (WHO 2024):

| Emergency Type | Prevalence/Urgency | Triage Priority |
|----------------|---------------------|-----------------|
| **Malaria (severe)** | Leading cause of morbidity. El Niño 2024 caused spikes. Children under 5 most affected | High — needs rapid assessment of severity (fever, convulsions, altered consciousness) |
| **Road traffic accidents** | Major cause of injury/death, especially on highways | High — assess for hemorrhage, head injury, fractures |
| **Maternal emergencies** | 260,000+ teen pregnancies/year. High maternal mortality | Critical — eclampsia, hemorrhage, obstructed labor |
| **Respiratory infections** | Pneumonia, TB prevalent | Medium-High — assess breathing difficulty, oxygen need |
| **Snakebite** | Common in rural/agricultural areas | High — identify snake type, assess envenomation signs |
| **Diarrheal diseases** | Especially in children, worsened by flooding | Medium — assess dehydration level |
| **Burns** | Common household injuries | Medium — assess burn area %, depth |
| **Assault/violence** | Urban areas | Variable — assess wound severity |

### Triage Protocol Recommendation

Use WHO Integrated Management of Childhood Illness (IMCI) and WHO Emergency Triage Assessment and Treatment (ETAT) protocols as the clinical backbone. These are:
- Designed for resource-limited settings
- Well-validated in Kenya specifically
- Algorithm-based (maps well to LLM-guided decision trees)
- Available in Swahili

---

## 6. MVP Architecture Recommendation

### Phase 1: Cloud-First MVP (Fastest to Build)

```
┌─────────────────────────────────────────────────────┐
│                    USER DEVICE                       │
│          Phone call / WhatsApp / Web app             │
└──────────────────────┬──────────────────────────────┘
                       │ Audio stream (WebRTC)
                       ▼
┌─────────────────────────────────────────────────────┐
│              PIPECAT VOICE PIPELINE                  │
│                                                      │
│  ┌──────────┐   ┌───────────┐   ┌────────────────┐ │
│  │ Whisper   │──▶│ Groq API  │──▶│ ElevenLabs     │ │
│  │ (STT)     │   │ (LLM)     │   │ Flash (TTS)    │ │
│  │ ~200ms    │   │ ~200ms    │   │ ~75ms          │ │
│  └──────────┘   └───────────┘   └────────────────┘ │
│                                                      │
│  System prompt: WHO ETAT triage protocols            │
│  Language: Swahili + English (code-switching)        │
└─────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│              TRIAGE OUTPUT                            │
│  - Emergency level (Red/Yellow/Green)                │
│  - Key symptoms identified                           │
│  - Recommended action + nearest facility             │
│  - Logged for clinical review                        │
└─────────────────────────────────────────────────────┘
```

### Model Selection for MVP

| Component | Primary | Fallback |
|-----------|---------|----------|
| **LLM** | GPT-4o-mini via API (best accuracy/speed ratio) | Groq + Llama-3-Meditron-8B (fastest, open-source medical) |
| **STT** | faster-whisper large-v3 (self-hosted, Swahili) | Google Cloud STT |
| **TTS** | ElevenLabs Flash v2.5 | Coqui XTTS v2 (offline) |
| **Framework** | Pipecat | LiveKit Agents |

### Cost Estimate (per conversation, ~3 minutes)

| Component | Cost |
|-----------|------|
| STT (Whisper self-hosted) | ~$0 (compute only) |
| LLM (GPT-4o-mini, ~2K tokens) | ~$0.001 |
| TTS (ElevenLabs, ~500 chars) | ~$0.09 |
| **Total per call** | **~$0.10** |

Using Groq + Coqui XTTS (all self-hosted except Groq): ~$0.01 per call.

---

## Sources

- [MedGemma on Hugging Face](https://huggingface.co/google/medgemma-4b-it)
- [MedGemma 1.5 on Hugging Face](https://huggingface.co/google/medgemma-1.5-4b-it)
- [Google Research: MedGemma](https://research.google/blog/medgemma-our-most-capable-open-models-for-health-ai-development/)
- [MedGemma — Google DeepMind](https://deepmind.google/models/gemma/medgemma/)
- [Meditron GitHub](https://github.com/epfLLM/meditron)
- [MEDITRON-70B Paper](https://arxiv.org/abs/2311.16079)
- [Llama-3-Meditron Paper](https://openreview.net/pdf?id=ZcD35zKujO)
- [BioMistral Paper](https://arxiv.org/abs/2402.10373)
- [BioMistral on Hugging Face](https://huggingface.co/BioMistral/BioMistral-7B)
- [OpenBioLLM (Saama)](https://huggingface.co/blog/aaditya/openbiollm)
- [Open Medical-LLM Leaderboard](https://huggingface.co/blog/leaderboard-medicalllm)
- [Med-Gemini Capabilities](https://arxiv.org/abs/2404.18416)
- [Google Med-Gemini Blog](https://research.google/blog/advancing-medical-ai-with-med-gemini/)
- [GPT-4 Medical Capabilities](https://arxiv.org/abs/2303.13375)
- [MedQA Benchmark](https://www.vals.ai/benchmarks/medqa)
- [Clinical Benchmarking of LLMs (2025)](https://www.medrxiv.org/content/10.64898/2025.12.29.25343145v1.full)
- [Best Inference Providers 2026](https://fast.io/resources/best-inference-providers-ai-agents/)
- [Inference Provider Comparison](https://infrabase.ai/blog/ai-inference-api-providers-compared)
- [Groq API Guide](https://markaicode.com/groq-api-fastest-llm-inference-python/)
- [Token Arbitrage: Provider Benchmarks](https://blog.gopenai.com/the-token-arbitrage-groq-vs-deepinfra-vs-cerebras-vs-fireworks-vs-hyperbolic-2025-benchmark-ccd3c2720cc8)
- [Pipecat GitHub](https://github.com/pipecat-ai/pipecat)
- [One-Second Voice Latency with Pipecat](https://modal.com/blog/low-latency-voice-bot)
- [Voice AI Agent Frameworks 2026](https://medium.com/@mahadise0011/top-voice-ai-agent-frameworks-in-2026-a-complete-guide-for-developers-4349d49dbd2b)
- [LiveKit Voice AI Agent Guide](https://www.f22labs.com/blogs/how-to-build-a-voice-ai-agent-using-livekit/)
- [6 Best Orchestration Tools for Voice Agents](https://www.assemblyai.com/blog/orchestration-tools-ai-voice-agents)
- [Voice AI Pipeline: 300ms Budget](https://www.channel.tel/blog/voice-ai-pipeline-stt-tts-latency-budget)
- [Bedrock vs LiveKit vs Pipecat Comparison](https://webrtc.ventures/2026/03/choosing-a-voice-ai-agent-production-framework/)
- [STT/TTS for Voice Agents Comparison](https://softcery.com/lab/how-to-choose-stt-tts-for-ai-voice-agents-in-2025-a-comprehensive-guide)
- [Deepgram vs ElevenLabs](https://deepgram.com/learn/deepgram-vs-elevenlabs)
- [Deepgram STT Benchmarks](https://deepgram.com/learn/speech-to-text-benchmarks)
- [Cartesia Sonic 3](https://cartesia.ai/sonic)
- [Coqui TTS GitHub](https://github.com/coqui-ai/tts)
- [Edge STT/TTS for Swahili (2025)](https://arxiv.org/abs/2510.16497)
- [Deepgram Language Support](https://developers.deepgram.com/docs/language)
- [Whisper v3 Large Guide](https://ucstrategies.com/news/whisper-v3-large-open-source-speech-to-text-guide-99-languages/)
- [Kenya Malaria (WHO 2024)](https://www.who.int/publications/m/item/malaria-2024-ken-country-profile)
- [Kenya Malaria Statistics](https://www.severemalaria.org/countries/kenya)
- [CDC in Kenya](https://www.cdc.gov/global-health/countries/kenya.html)
- [Medical LLMs Enterprise Guide](https://picovoice.ai/blog/medical-language-models-guide/)
- [Me-LLaMA Paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC11142305/)
- [llama.cpp GitHub](https://github.com/ggml-org/llama.cpp)
- [GGUF Quantization Guide](https://towardsdatascience.com/quantize-llama-models-with-ggml-and-llama-cpp-3612dfbcc172/)
