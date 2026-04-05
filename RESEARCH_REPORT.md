# Voice-Based AI Medical Triage System: Architecture Research Report

**Date:** April 2026  
**Focus:** MVP for Kenya

---

## 1. Existing Medical Triage AI Systems

### Corti AI (Emergency Dispatch)
- **What it does:** Listens to live emergency dispatch calls and detects critical illness in real-time using ML models
- **Architecture:** The Corti Orb is an edge device that sits on a dispatcher's desk, runs ML models on Arm processors, and works even without internet. Real-time audio analysis detects patterns indicating specific emergencies (e.g., cardiac arrest)
- **2025 update:** Launched specialized healthcare foundation models -- up to 20 expert models functioning like healthcare specialists for coding, quality control, summarization
- **Relevance:** Proves that real-time voice analysis for medical emergencies works in production. Edge-first design is relevant for low-connectivity environments like Kenya

### Babylon Health
- **Architecture:** Hybrid model combining Bayesian inference + ML, built on a massive medical knowledge graph using SNOMED CT ontology. Uses NLP to interpret symptoms, linked to a structured medical knowledge base. Deep reinforcement learning optimizes decision-making flows
- **Africa deployment:** Active in Rwanda and other African markets. Users describe symptoms via voice or text in local languages; system provides risk assessments and care recommendations
- **Status:** Company went through financial difficulties; technology still relevant as reference architecture

### Ada Health
- **Architecture:** Probabilistic + rule-based reasoning engine that simulates how a doctor narrows down diagnoses through structured questioning. Conversational interface where users input symptoms and receive possible conditions, severity insights, and next steps
- **Relevance:** Text-based triage flow that could be adapted for voice

### AI Triage Deployments in Africa
- **Kenya:** AI diagnostic platforms deployed to identify pneumonia in children, reducing referral delays
- **Ethiopia:** HEP Assist (by Ministry of Health + Last Mile Health + IDinsight) -- AI tool guiding frontline health workers through triage and referral decisions in real-time
- **Rwanda:** AI-powered portable X-ray machines screening up to 300 people/day in remote areas
- **Senegal:** AI decision support integrated into CHW programs; correct referral rates increased, unnecessary hospital visits decreased
- **Key insight:** AI-assisted triage saves patients over 13 minutes of treatment time (2025 study). Systems must accommodate low bandwidth, diverse accents, and local health data

---

## 2. Voice AI Agent Frameworks

### Pipecat (by Daily) -- RECOMMENDED FOR MVP
- **Type:** Open-source Python framework (Apache 2.0)
- **Architecture:** Pipeline-based -- series of processors handling real-time audio, text, and video frames. Client (browser/mobile/phone) connects to server running the Pipecat pipeline
- **Transport:** WebRTC (via Daily) for ultra-low latency; also supports WebSocket
- **Key features:**
  - 40+ AI service integrations as plugins
  - SDKs for Python, JavaScript, React, iOS, Android, C++
  - Silero VAD for voice activity detection
  - Streaming STT -> LLM -> TTS pipeline built-in
  - Modular and composable architecture
- **Telephony:** Native Twilio integration via WebSocket Media Streams. Supports both dial-in (user calls bot) and dial-out (bot calls user)
- **Latency:** Sub-1-second voice-to-voice demonstrated (Modal + Pipecat benchmark)
- **Pricing:** Free (open source). You pay for the AI services you plug in (STT, LLM, TTS) and hosting
- **Medical triage fit:** Excellent. Full control over conversation flow, can encode triage protocols as pipeline logic, integrates with any LLM

### LiveKit Agents -- STRONG ALTERNATIVE
- **Type:** Open-source framework (Apache 2.0) built on LiveKit WebRTC server
- **Architecture:** Agent code in Python; LiveKit handles WebRTC transport, room management, scaling
- **Key features:**
  - Semantic turn detector with sub-75ms P99 latency
  - Full open-source stack, self-hostable
  - Voice, video, and text modalities
  - HIPAA-eligible on Scale tier ($500/month)
- **Pricing:** Self-hosted = free (your infra costs only). Cloud: $0.01/min agent session + $0.004/min audio
- **Medical triage fit:** Very good. Self-hosting eliminates data residency concerns. WebRTC transport handles poor networks well

### Vapi -- MANAGED OPTION
- **Type:** Closed-source managed voice AI platform
- **Architecture:** Modular -- orchestrates STT, LLM, TTS behind an API. Bring Your Own Telephony (Twilio, Telnyx)
- **Latency:** Sub-500ms round-trip consistently
- **Pricing:** $0.05/min orchestration + STT ($0.01/min) + LLM ($0.02-0.20/min) + TTS ($0.04/min) + telephony ($0.01/min). Total: $0.07-$0.25/min. Annual budget typically $40K-$70K for production
- **Medical triage fit:** Fast to prototype but expensive at scale. Less control over conversation logic

### Retell AI -- FASTEST TO PROTOTYPE
- **Type:** Managed voice AI platform with visual workflow builder
- **Architecture:** WebSocket-based real-time streaming. Dynamic variables for multi-tenant deployments
- **Latency:** Low latency, natural-sounding conversations (specific ms not published)
- **Pricing:** $0.07+/min pay-as-you-go, no platform fee. $10 free credits to start (~60 min)
- **HIPAA/SOC2 support:** Yes
- **Medical triage fit:** Already used for medical clinic AI receptionists. Visual workflow builder is fast for prototyping. Less control than open-source options
- **Time to first call:** ~3 hours

### Vocode -- LIGHTWEIGHT OPTION
- **Type:** Open-source (YC-backed), Python library
- **Architecture:** Orchestrates STT + LLM + TTS with real-time streaming. Supports phone calls, Zoom, web
- **Key features:** Handles endpointing, interruptions, turn-taking. 10-line integration possible
- **Medical triage fit:** Good for rapid prototyping. Less mature ecosystem than Pipecat/LiveKit
- **Status:** Smaller community than Pipecat; development pace uncertain

### Comparison Summary

| Feature | Pipecat | LiveKit | Vapi | Retell | Vocode |
|---------|---------|---------|------|--------|--------|
| Open Source | Yes | Yes | No | No | Yes |
| Self-Hostable | Yes | Yes | No | No | Partial |
| Twilio Integration | Yes | Yes | Yes | Yes | Yes |
| Time to Prototype | Days | Days | Hours | Hours | Days |
| Cost at Scale | Low | Low | High | Medium | Low |
| Control over Logic | Full | Full | Limited | Limited | Full |
| Community Size | Large | Large | N/A | N/A | Small |
| HIPAA Ready | DIY | Yes ($500/mo) | No | Yes | No |

---

## 3. End-to-End Architecture Patterns

### The Voice AI Pipeline

```
[Caller's Phone] 
    --> [Telephony: Twilio / Africa's Talking]
    --> [Transport: WebSocket or WebRTC]
    --> [Voice Activity Detection (VAD)]
    --> [Streaming STT: Deepgram / Whisper]
    --> [LLM: GPT-4o / Claude / Llama]
    --> [Streaming TTS: ElevenLabs / Deepgram / Cartesia]
    --> [Transport back to caller]
```

### Two Architecture Approaches

**Cascading (STT -> LLM -> TTS):**
- Traditional pipeline, most flexible
- Each component independently swappable
- Latency budget: ~300ms total target
  - STT finalization: ~50-100ms
  - LLM time-to-first-token: ~100-150ms
  - TTS time-to-first-byte: ~50-100ms
  - Transport overhead: ~30-50ms
- Total achievable: **500-800ms** with good optimization
- **Recommended for MVP** -- more control, easier to debug

**Speech-to-Speech (Multimodal):**
- Single model processes audio directly (e.g., OpenAI Realtime API, Gemini Live)
- Lower latency potential (~300ms)
- Less control over conversation flow
- Higher cost per minute
- Not recommended for medical triage (need structured logic)

### WebSocket vs WebRTC

**WebSocket (Recommended for phone integration):**
- Simpler to implement
- Works well with Twilio Media Streams
- TCP-based -- reliable but slightly higher latency
- Sufficient for phone-quality audio

**WebRTC (Recommended for web/mobile app):**
- UDP-based -- lowest possible latency
- Better handling of poor network conditions
- Built-in echo cancellation, noise suppression
- More complex to implement
- LiveKit and Daily provide managed WebRTC infrastructure

### Handling Interruptions (Barge-In)

Critical for medical triage where callers may be distressed:

1. **Voice Activity Detection (VAD):** Continuously monitor for speech during AI output. Use Silero VAD with 300-500ms silence threshold
2. **Immediate stop:** When interruption detected, stop TTS playback within 200ms
3. **Context preservation:** Buffer what the AI was saying; resume or pivot based on caller input
4. **Smart filtering:** Distinguish genuine interruptions from backchannels ("mm-hmm", "yeah", coughs, background noise)
5. **Advanced:** Prosodic signals (intonation, pitch) and lexical cues help anticipate turn ends

### Turn-Taking Design for Medical Triage

- Use longer silence thresholds (500-800ms) for medical contexts -- callers may need time to describe symptoms
- Implement "thinking" indicators (brief audio cue) so callers know the system is processing
- Allow the system to ask "Are you still there?" after extended silence
- For critical questions (e.g., "Is the patient breathing?"), wait longer before timing out

---

## 4. Medical Triage Protocols

### MPDS (Medical Priority Dispatch System) -- MOST RELEVANT
- **Gold standard** for emergency medical dispatching for 40+ years
- **36 validated protocols** covering all major complaint types
- **Encoding format:** Three-component code (Number-Letter-Number)
  - First number (1-36): Complaint/protocol type
  - Letter (A-E, Omega): Severity level (A=lowest, E=highest, Omega=life-threatening)
  - Second number: Sub-determinant with specific condition details
  - Example: 9-E-1 = Cardiac arrest, not breathing
  - Example: 3-A-3 = Superficial animal bite
- **Software version:** ProQA is the computerized version, integrates with CAD systems
- **How to encode in AI:** Structure LLM prompts to follow MPDS flowcharts. Each protocol is a decision tree that can be represented as a state machine. The AI asks scripted questions in sequence, classifies responses, and determines the determinant code

### Manchester Triage System (MTS)
- Used in emergency departments (not dispatch)
- Flowchart-based with discriminators for each presenting complaint
- Classifies into 5 urgency levels (1=immediate to 5=non-urgent)
- Well-suited for AI encoding as decision trees
- **Not used for mass casualty or phone triage** -- more relevant for in-person ED triage

### START Triage (Simple Triage and Rapid Treatment)
- Designed for mass casualty incidents
- Simple algorithm: Can patient walk? Breathing? Pulse? Mental status?
- Categories: Immediate (Red), Delayed (Yellow), Minor (Green), Deceased (Black)
- Very simple to encode as a decision tree
- Less relevant for phone-based triage

### Recommended Approach for AI Encoding

1. **Use MPDS as the primary framework** -- it's designed for phone-based triage
2. **Encode each of the 36 protocols as a state machine** with:
   - Entry criteria (chief complaint matching)
   - Sequential question nodes
   - Branching logic based on responses
   - Determinant code output
   - Pre-arrival instructions
3. **Use the LLM for:**
   - Natural language understanding of caller responses (mapping free-form speech to protocol answers)
   - Generating empathetic, clear questions in the caller's language
   - Handling edge cases and unexpected responses
4. **Do NOT use the LLM for:**
   - Making triage decisions (protocol logic should be deterministic)
   - Providing medical diagnoses
   - Overriding protocol-determined severity levels

---

## 5. Regulatory and Safety Considerations (Kenya)

### Regulatory Framework
- **Data Protection Act 2019** -- Primary data privacy law. Requires consent for processing personal data, data minimization, purpose limitation
- **Digital Health Act 2023** -- Governs digital health technologies
- **Digital Health (Health Information Management Procedures) Regulations 2025** -- New regulations for health data management
- **Kenya AI Strategy 2025-2030** -- Positions Kenya as regional AI leader, but **no AI-specific enforceable legislation yet**
- **No specific medical AI regulation** -- falls under general medical device and data protection frameworks
- **Key risk:** Unclear regulatory direction; an AI Bill is being discussed, creating uncertainty

### Required Safety Guardrails

1. **Always recommend calling emergency services** for life-threatening conditions
   - System should say: "Based on what you've described, please call [emergency number] immediately"
   - Provide Kenya's emergency numbers: 999, 112, or specific ambulance services
   
2. **Never provide definitive diagnoses**
   - Frame all output as "possible conditions" or "this may indicate"
   - Always recommend professional medical evaluation
   - Include disclaimer at start of every call

3. **Escalation protocols**
   - Automatic escalation to human operator for high-severity cases
   - Clear handoff when system confidence is low
   - Log all interactions for clinical review

4. **Liability disclaimers**
   - Explicit verbal disclaimer at call start
   - Caller acknowledgment before proceeding
   - Record consent

### Data Privacy Requirements

1. **Consent:** Must obtain explicit consent before recording medical conversations
2. **Data minimization:** Only collect information necessary for triage
3. **Storage:** Health data must be stored securely; consider local data residency
4. **Retention:** Define and enforce data retention periods
5. **Access:** Patients must be able to access their data
6. **Encryption:** End-to-end encryption for voice data in transit and at rest
7. **De-identification:** Remove PII from data used for model training

### Practical Implications
- Self-hosting (LiveKit/Pipecat) gives you full control over data residency in Kenya
- Avoid sending voice data to US/EU-hosted services if possible
- Keep audio recordings encrypted and access-controlled
- Implement audit logging for all data access

---

## 6. Tech Stack Recommendations for MVP

### Fastest Path to Working Prototype

#### Option A: Managed Platform (2-3 weeks to demo)
```
Phone Call --> Twilio Voice --> Retell AI / Vapi
                                  |
                              GPT-4o (LLM)
                                  |
                         Triage logic in prompts
                                  |
                         Response back to caller
```
- **Pros:** Fastest to build, hosted infrastructure, built-in telephony
- **Cons:** Expensive at scale ($0.07-0.25/min), less control, data leaves your infrastructure
- **Best for:** Quick demo to stakeholders, validating concept

#### Option B: Open-Source Stack (4-6 weeks to MVP) -- RECOMMENDED
```
Phone Call --> Africa's Talking Voice API (or Twilio)
                     |
              Pipecat Pipeline (self-hosted)
                     |
    [Silero VAD] --> [Deepgram STT] --> [Claude/GPT-4o] --> [Deepgram TTS]
                     |
              Triage State Machine
                     |
         Response streamed back to caller
```
- **Pros:** Full control, lower cost at scale, data stays in your infra, customizable triage logic
- **Cons:** More engineering effort, need to manage infrastructure
- **Best for:** Production system, medical compliance

#### Option C: Hybrid Quick-Start (1-2 weeks to demo, upgradeable)
```
Web App (React) --> LiveKit WebRTC --> LiveKit Agent (Python)
                                           |
                     [Deepgram STT] --> [Claude API] --> [ElevenLabs TTS]
                                           |
                                    Triage Protocol Logic
```
- **Pros:** Web-based (no telephony complexity), upgradeable to phone later, good latency
- **Cons:** Requires internet-connected device with browser
- **Best for:** Starting with web demo, adding phone later

### Phone-Based vs Web-Based vs Mobile

| Factor | Phone (Twilio/AT) | Web App | Mobile App |
|--------|-------------------|---------|------------|
| Reach in Kenya | Highest (any phone) | Medium (needs internet) | Lower (needs smartphone) |
| Setup Complexity | Medium | Low | High |
| Latency | Good (WebSocket) | Best (WebRTC) | Best (WebRTC) |
| Cost per Interaction | Higher (telephony fees) | Lowest | Lowest |
| Offline Capability | Works on feature phones | No | Possible |
| Time to MVP | 4-6 weeks | 2-3 weeks | 8-12 weeks |

### Africa's Talking vs Twilio for Kenya

**Africa's Talking (Recommended for Kenya production):**
- Built for Africa -- local routing infrastructure in 30+ countries
- USSD support (critical for feature phones)
- Lower cost for African countries
- Voice API supports inbound/outbound calls
- SMS fallback for triage results
- **Limitation:** Less documentation, smaller developer community

**Twilio (Recommended for prototype):**
- Better documentation, larger community
- Easier integration with Pipecat (native support)
- Media Streams WebSocket API well-documented
- More expensive for Kenya calls
- **Use for prototype, migrate to Africa's Talking for production**

### STT for Swahili/Kenyan English

- **Deepgram Whisper Cloud:** Supports Swahili. Best balance of speed, accuracy, and cost
- **OpenAI Whisper API:** Supports Swahili (trained on ~5.4 hours of Swahili data -- limited but functional)
- **Google Cloud STT:** Good Swahili support, higher latency
- **Recommendation:** Start with Deepgram for English, test Whisper for Swahili. Plan for accuracy limitations in Swahili and Kenyan-accented English

### Recommended MVP Tech Stack

```
LAYER              TECHNOLOGY              COST (ESTIMATE)
---------------------------------------------------------------
Telephony          Twilio (prototype)      ~$0.02/min
                   Africa's Talking (prod) ~$0.01/min

Transport          Pipecat + WebSocket     Free (open source)

VAD                Silero VAD              Free (open source)

STT                Deepgram Nova-2         $0.0043/min
                   (+ Whisper for Swahili) $0.0048/min

LLM                Claude 3.5 Sonnet       ~$0.01-0.03/min
                   (or GPT-4o-mini)        ~$0.005/min

TTS                Deepgram Aura           $0.0050/min
                   (or ElevenLabs)         $0.015/min

Triage Logic       Python state machine    Free

Hosting            AWS/GCP (Nairobi        ~$50-200/mo
                   region or closest)

Database           PostgreSQL              Included in hosting
---------------------------------------------------------------
TOTAL PER MINUTE (ESTIMATED):             $0.04 - $0.08/min
```

### MVP Development Roadmap

**Week 1-2: Core Pipeline**
- Set up Pipecat with Deepgram STT + Claude + Deepgram TTS
- Build basic conversation loop on web (WebRTC via Daily)
- Test end-to-end latency

**Week 3-4: Triage Protocol**
- Encode 5 most common MPDS protocols as state machines
- Integrate triage logic with LLM conversation flow
- Add safety guardrails and disclaimers

**Week 5-6: Telephony Integration**
- Connect Pipecat to Twilio for phone-based access
- Test with Kenyan phone numbers
- Handle poor audio quality, background noise

**Week 7-8: Testing and Refinement**
- Test with Kenyan English and Swahili speakers
- Tune VAD and silence thresholds for medical context
- Add SMS follow-up with triage results via Africa's Talking
- User testing with healthcare workers

---

## Key Recommendations

1. **Start with Pipecat + Deepgram + Claude** -- best balance of control, cost, and speed
2. **Build web demo first** (2 weeks), then add phone integration (2 more weeks)
3. **Encode triage protocols as deterministic state machines** -- use the LLM for NLU and response generation only, not for triage decisions
4. **Plan for Swahili from day one** even if English MVP comes first -- architecture must support multilingual
5. **Self-host everything possible** to maintain data residency in Kenya
6. **Africa's Talking for production telephony** -- Twilio for prototyping
7. **Always include safety rails** -- emergency number recommendations, disclaimers, human escalation paths
8. **Target <800ms end-to-end latency** -- achievable with streaming pipeline

---

## Sources

### Existing Medical AI Systems
- [Corti AI Healthcare Infrastructure](https://www.corti.ai/news/corti-launches-specialized-healthcare-ai-infrastructure-challenging-industry-misuse-of-general-purpose-models)
- [EENA & Corti AI Emergency Dispatch](https://eena.org/knowledge-hub/press-releases/artificial-intelligence-eena-corti-project/)
- [Babylon Health AI](https://www.babylonhealth.com/ai)
- [Babylon Health AI-Driven Diagnostics](https://headofai.ai/ai-industry-case-studies/babylon-health-ai-driven-healthcare-diagnostics/)
- [Multilingual AI Triage for Africa](https://pmc.ncbi.nlm.nih.gov/articles/PMC12344245/)
- [AI Agents Transform Healthcare Across Africa](https://www.iqvia.com/locations/middle-east-and-africa/blogs/2025/11/how-ai-agents-can-transform-healthcare-across-africa)
- [AI-Native Triage for Africa's Hospitals](https://pctechmag.com/2026/03/can-ai-native-triage-relieve-pressure-on-africas-overburdened-public-hospitals/)
- [African AI Healthtech Firms](https://african.business/2025/05/technology-information/the-african-ai-healthtech-firms-saving-lives-and-attracting-investors)

### Voice AI Frameworks
- [Pipecat GitHub](https://github.com/pipecat-ai/pipecat)
- [Pipecat Documentation](https://docs.pipecat.ai/getting-started/introduction)
- [Pipecat Review (Neuphonic)](https://www.neuphonic.com/blog/pipecat-review-open-source-ai-voice-agents)
- [One-Second Latency with Pipecat (Modal)](https://modal.com/blog/low-latency-voice-bot)
- [LiveKit Agents GitHub](https://github.com/livekit/agents)
- [LiveKit Pricing](https://livekit.com/pricing)
- [LiveKit vs Vapi Comparison](https://modal.com/blog/livekit-vs-vapi-article)
- [LiveKit Review (Neuphonic)](https://www.neuphonic.com/blog/livekit-review-open-source-webrtc-ai-voice-tool)
- [Vocode GitHub](https://github.com/vocodedev/vocode-core)
- [Vapi AI Review](https://synthflow.ai/blog/vapi-ai-review)
- [Retell AI Review](https://synthflow.ai/blog/retell-ai-review)
- [Voice Agent Platform Comparison (Softcery)](https://softcery.com/lab/choosing-the-right-voice-agent-platform-in-2025)
- [Best Voice Agent Stack (Hamming)](https://hamming.ai/resources/best-voice-agent-stack)

### Architecture & Latency
- [Voice AI Pipeline 300ms Budget](https://www.channel.tel/blog/voice-ai-pipeline-stt-tts-latency-budget)
- [Real-Time vs Turn-Based Architecture](https://softcery.com/lab/ai-voice-agents-real-time-vs-turn-based-tts-stt-architecture)
- [Voice AI Workflows STT+NLP+TTS (Deepgram)](https://deepgram.com/learn/designing-voice-ai-workflows-using-stt-nlp-tts)
- [Barge-In Detection (SparkCo)](https://sparkco.ai/blog/optimizing-voice-agent-barge-in-detection-for-2025)
- [Adaptive Interruption Handling (LiveKit)](https://livekit.com/blog/adaptive-interruption-handling)

### Triage Protocols
- [MPDS Overview](https://www.emergencydispatch.org/what-we-do/emergency-priority-dispatch-system/medical-protocol)
- [MPDS Wikipedia](https://en.wikipedia.org/wiki/Medical_Priority_Dispatch_System)
- [AI-Powered Triage in EDs 2026](https://www.iatrox.com/blog/ai-powered-triage-2026-emergency-departments-machine-learning)

### Kenya Regulatory & Infrastructure
- [Kenya AI Strategy 2025-2030](https://www.insideprivacy.com/artificial-intelligence/kenyas-ai-strategy-2025-2030-signals-for-global-companies-operating-in-africa/)
- [Kenya Data Privacy Laws for Clinics](https://www.easyclinic.io/data-privacy-laws-in-kenya/)
- [Kenya Digital Health Regulations 2025](https://new.kenyalaw.org/akn/ke/act/ln/2025/76/eng@2025-04-11)
- [AI in Health: Policy Pathways for Kenya](https://cipit.strathmore.edu/ai-in-health-highlights-and-policy-pathways-for-kenyas-healthcare-future/)
- [Africa's Talking APIs](https://africastalking.com/)
- [Africa's Talking Voice API](https://africastalking.com/voice/)
- [Africa's Talking vs Twilio](https://www.courier.com/integrations/compare/africas-talking-vs-twilio)

### STT for African Languages
- [Deepgram Languages Overview](https://developers.deepgram.com/docs/models-languages-overview)
- [Benchmarking ASR for African Languages](https://arxiv.org/html/2512.10968)
- [Deepgram Multilingual Guide](https://deepgram.com/learn/multilingual-speech-to-text-guide)
