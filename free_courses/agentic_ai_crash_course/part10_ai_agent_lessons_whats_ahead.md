# Part 10: AI Agent Lessons and What's Ahead


## A Quick Recap

Here’s what we covered over the last 9 parts:

- **Part 1 — What agents are:** Not just chatbots that generate text, but systems that can decide and act.  
- **Part 2 — Types of agents:** From tightly controlled workflow agents to fully autonomous ones, depending on how much decision-making you hand over.  
- **Part 3–4 — Tools and RAG:** The bread and butter of agent action and knowledge grounding.  
- **Part 5 — MCP:** A clean way to structure everything an agent needs (tools, memory, prior messages) into one payload.  
- **Part 6 — Planning and reasoning models:** Why plain LLMs aren’t enough for complex decisions, and how newer models are built for multi-step tasks.  
- **Part 7 — Memory:** Short-term vs. long-term memory, what to store, how to retrieve, and why it matters for continuity.  
- **Part 8 — Multi-agent systems:** Orchestration, peer-to-peer collaboration, and the messiness of coordination.  
- **Part 9 — Real-world systems:** How Perplexity, NotebookLM, and DeepResearch likely use these patterns in different ways.

We’ve covered the **moving parts** that show up in real-world systems.  
But all of it falls apart if you’re not thinking about two things: **observability** and **evaluation**.

---

## What’s Still Hard

### Observability
Observability means tracking what your agent is doing — at every step. You’ll want:
- Logs of tool calls, decisions, retries  
- Metrics to spot bottlenecks in latency and cost  
- Visibility into when things go off-rail  
- Step-wise traceability for debugging  

Tools like **Comet Opik** help with this.  
Design observability **from day one**, especially for high-autonomy agents.

---

### Evaluation
Agents are **non-deterministic**.  
You need **continuous evaluation**, not just manual testing.

At a minimum, track:
- Goal or task completion rates  
- Tool call success/failure  
- RAG quality and hallucination metrics  
- Model overthinking or inefficiency  
- Latency and token usage at each step  

Evaluation is how you **understand** and **improve** your system.  
Too many teams do *vibe checks* instead of real evals — and get stuck in **PoC purgatory**.

Think of evals + observability as your **testing pipeline** — the agentic equivalent of software QA.  
Metrics will vary by use case, but the discipline is the same.

---

## Where Things Are Headed in Agentic AI

This space is early, but here are clear trends:

---

### 1. Protocols > Prompts
<img width="800" height="800" alt="image" src="https://github.com/user-attachments/assets/13b78c2e-fc2b-41fd-a0e5-aa88c60187ea" />

_Image Source: Reuven’s LinkedIn post_

As systems grow, we’ll move away from handcrafted prompts toward shared **standards**.

- **MCP** (Model Context Protocol) standardizes how we package structured context — tools, memory, RAG, prior instructions.  
- **A2A** (Agent-to-Agent), released by Google, focuses on cross-platform agent communication with a shared schema.

Expect cleaner abstractions over time — though it’ll take a while before anything becomes as standard as HTTP.

---

### 2. Hybrid Reasoning Models
Reasoning models will evolve toward **selective planning** — knowing when to plan vs. act fast.

We’re already seeing this with **Claude 3.7** and others.  
The aim: balance intelligence with efficiency — without overthinking every task.

---

### 3. Better Memory Systems
Today’s memory is mostly **duct-taped in**.  
The future: memory that knows **what to recall, when, and why**.  
Expect:
- Task-scoped memory  
- Session-based memory  
- Persona-specific memory  

And **easier management**.

---

### 4. Tool Ecosystem Maturity
Right now, everyone’s building custom tools/wrappers. Over time:
- Trusted, plug-and-play APIs  
- Better abstraction layers  
- Shared security practices  

Just like microservices matured in traditional software, tools will mature in the **agentic stack**.

---

## Latest Developments (2024-2025 Update)

### MCP Goes Mainstream

**Model Context Protocol** has achieved industry-wide adoption:

- **OpenAI** (March 2025): Integrated across ChatGPT desktop, Agents SDK, and Responses API
- **Microsoft** (May 2025): General availability in Copilot Studio with enhanced tracing and streaming
- **Google**: Support in Vertex AI and Gemini API
- **12+ Major SDKs**: Claude Agent SDK, LangChain, LlamaIndex, CrewAI, and more

MCP is now the de facto standard for agent-system communication, solving the fragmentation problem where every framework had different tool integration approaches.

**A2A Evolution:**
LangChain's Agent-to-Agent Protocol is gaining traction for inter-agent communication, complementing MCP's agent-to-system focus.

---

### Reasoning Models Transform Agents

**DeepSeek-R1** (January 2025) and **OpenAI o3** marked a breakthrough in agent capabilities:

**DeepSeek-R1:**
- First open-source reasoning model trained via pure RL
- Performance comparable to o1 at 96% lower cost
- Training cost: $294K (vs. millions for traditional models)
- Open weights enabling widespread research

**Impact on Agents:**
- Multi-step planning without manual prompt engineering
- Self-verification reducing hallucinations
- Dynamic strategy adaptation based on task complexity
- Better handling of complex, multi-hop reasoning tasks

**Reasoning models are becoming the brains of next-gen agentic systems.**

---

### Production Agent Frameworks Mature

**2025 marks the shift from prototypes to production:**

**Leading Frameworks:**
- **LangGraph**: Graph-based orchestration for deterministic workflows, adopted in regulated industries
- **LlamaIndex Workflows 1.0**: Production-ready data pipelines for RAG-heavy agents
- **CrewAI**: Simplified multi-agent collaboration with role-based design
- **Claude Agent SDK**: Security-first, production-ready agentic applications

**Key Trends:**
- Focus on observability and debugging built-in
- Standardized evaluation frameworks (AgentBench, WebArena, GAIA)
- Human-in-the-loop patterns for critical decisions
- Cost management and token tracking as first-class features

---

### Real-World Agent Deployments

**Production systems proving agent value:**

- **GitHub Copilot Workspace**: Multi-agent system for full-stack development
- **Replit Agent**: Autonomous full-stack application development
- **Perplexity Pro**: Agentic research with real-time web access
- **Google NotebookLM**: Document analysis and synthesis
- **Customer Support**: Automated ticket routing and resolution at scale

**Common Patterns:**
- Hybrid human-AI workflows
- Specialized agents for different domains
- Extensive guardrails and safety checks
- Continuous monitoring and evaluation

---

### Multimodal Agents on the Horizon

**2025 sees the rise of natively multimodal agents:**

**Key Developments:**
- **Gemini 2.5**: Multimodal Live API for real-time audio/video agent interactions
- **LLaMA 4**: Natively multimodal models (10M tokens) for document-heavy workflows
- **Vision-Language Agents**: Analyzing images, videos, and documents together

**New Capabilities:**
- Screen understanding for UI automation
- Document analysis with figures and tables
- Video comprehension for surveillance and analysis
- Robotics integration with visual feedback

---

### The Economic Shift

**DeepSeek's Impact:**

DeepSeek proved that SOTA agents don't require massive budgets:
- 96% cost reduction vs. proprietary reasoning models
- Open weights democratizing agent capabilities
- Training costs dropping from millions to hundreds of thousands

**Implications:**
- Smaller organizations can now build advanced agents
- Rapid innovation through open research
- More startups entering the agentic AI space
- Pressure on proprietary providers to reduce costs

---

### What to Watch in 2025-2026

**Emerging Trends:**

1. **Agentic RAG Becomes Standard**: Self-organizing retrieval with multi-source synthesis
2. **Specialized Agent Networks**: Domain-specific agents collaborating on complex tasks
3. **Continuous Learning**: Agents that improve from user interactions
4. **Cross-Platform Agents**: Operating seamlessly across tools and platforms via MCP
5. **Explainable Agents**: Better transparency into agent decision-making
6. **Edge Agents**: Running sophisticated agents on-device for privacy

**Technical Challenges Being Solved:**
- Hallucination detection and mitigation in multi-step workflows
- Efficient memory management for long-running agents
- Secure tool execution and sandboxing
- Cost-effective scaling of agent systems
- Reliable evaluation frameworks

---

## Resources for Continued Learning

**Key Papers (2024-2025):**
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)
- [GraphRAG: Knowledge Graphs for RAG](https://github.com/microsoft/graphrag)
- [Survey on Large Language Model Based Autonomous Agents](https://arxiv.org/abs/2308.11432)

**Frameworks to Explore:**
- [LangChain](https://www.langchain.com/) & [LangGraph](https://www.langchain.com/langgraph)
- [LlamaIndex](https://www.llamaindex.ai/)
- [CrewAI](https://www.crewai.com/)
- [Claude Agent SDK](https://www.anthropic.com/)

**Evaluation Tools:**
- [RAGAS](https://docs.ragas.io/) - RAG evaluation
- [AgentBench](https://github.com/THUDM/AgentBench) - Agent benchmarking
- [Comet Opik](https://www.comet.ml/docs/opik/) - LLM observability

---

## A Final Word

If you’ve followed along, you’ve seen the theme:

We didn’t start with **architecture**.  
We started with **problems**.

That’s the real mindset shift:  
> Don’t chase agents for the hype.  
> Build them when they make solving a problem easier, faster, or smarter.

**Start simple. Measure everything. Scale when needed.**  
Agent-first thinking breaks. Problem-first thinking scales.

---

Thanks for reading, sharing, and thinking along during these 10 parts.  
If you take away one thing from this series — let it be this:

> **Problem first, always.**

Check out the readme for more lectures and advanced topics. If this was helpful, feel free to forward it to someone looking to learn in this space. And if you’d like to go deeper, our full 6-week course covers system design, applied agentic concepts, and real evaluation workflows, the kind that support production-grade applications. The course is built for everyone, whether you’re a Product Manager, Architect, Director, C-suite leader, or someone seriously exploring agentic AI.

Our next cohort starts soon. Our next cohort starts soon. Early bird pricing is live: use the code "GITHUB" to get $300 off (Valid only for August 2025) to [register here](https://maven.com/aishwarya-kiriti/genai-system-design)!!

