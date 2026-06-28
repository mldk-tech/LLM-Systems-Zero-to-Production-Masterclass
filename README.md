# LLM Systems: Zero to Production MasterclassAuthor  
Level: Expert / MLOps Engineer

## Phase 1: Architecture and Foundations
1. 30-Second OverviewModern language models (from LLaMA-3 to Mistral-7B) are based on the Transformer architecture (Decoder-only). The beating heart of the architecture is the Self-Attention mechanism, which allows the model to weigh the importance of each token in relation to all other tokens in the context.
2. Mathematics: Scaled Dot-Product AttentionThe attention mechanism is defined as:$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$Where:$Q$ (Queries), $K$ (Keys), $V$ (Values) are matrices obtained by multiplying the input by learned weights.$d_k$ is the dimension of the Key (used for normalization to prevent extreme gradient values in the softmax).The Bottleneck: The $QK^T$ operation requires memory and time complexity of $O(n^2)$ (where $n$ is the sequence length). Therefore, Flash Attention was developed, which computes this equation in blocks (Tiling) on the GPU's fast SRAM.
3. Implementation Code (PyTorch)Here is a basic implementation of Self-Attention, plus a Causal Mask (so the model doesn't look into the future):

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Projections for Q, K, V
        self.c_attn = nn.Linear(d_model, 3 * d_model)
        self.c_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, C = x.size() # Batch, Time (seq_len), Channels (d_model)
        
        # Calculate Q, K, V
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)
        
        # Reshape for multi-head: (B, num_heads, T, d_k)
        q = q.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        
        # Efficient Attention (uses Flash Attention if supported by hardware/PyTorch version)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        
        # Reassemble
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

# Example: Batch of 2, 128 tokens, 768 dimensions, 12 heads
x = torch.randn(2, 128, 768)
attn = CausalSelfAttention(768, 12)
output = attn(x)
print(f"Output shape: {output.shape}") # (2, 128, 768)
```

4. Trade-offs & BenchmarksGQA (Grouped Query Attention) vs. MHA (Multi-Head): LLaMA-3 uses GQA. By sharing heads of $K$ and $V$, we significantly reduce the size of the KV Cache (by 4-8x), at a negligible cost of a drop in Perplexity (less than 1%).Paper to Read: FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness (Dao et al., 2022).

## Phase 2: Training and Optimization (Training & Fine-Tuning)

1. 30-Second OverviewWe do not train a foundation model from scratch (requires thousands of H100s). We will use an existing model (e.g., Mistral-7B or LLaMA-3.1-8B) and perform Supervised Fine-Tuning (SFT) on it, followed by Alignment using DPO to adapt it to our business task. We will use LoRA to train only a fraction of the weights and avoid Catastrophic Forgetting.
2. Mathematics: LoRA (Low-Rank Adaptation)Instead of updating the original weight matrix $W \in \mathbb{R}^{d \times k}$, we freeze $W$ and learn a low-rank update:$$ W_{new} = W + \Delta W = W + BA $$
Where $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$, and the rank $r \ll \min(d,k)$. Usually $r=16$.3. Implementation Code (HuggingFace PEFT)

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

model_id = "meta-llama/Llama-3.1-8B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
# Load model in 4-bit precision to save VRAM (requires bitsandbytes)
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    load_in_4bit=True, 
    device_map="auto"
)

# Configure LoRA
lora_config = LoraConfig(
    r=16, # Rank
    lora_alpha=32, # Scaling factor
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], # Apply to attention modules
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Wrap the model
peft_model = get_peft_model(model, lora_config)
peft_model.print_trainable_parameters()
# Output example: trainable params: 13,631,488 || all params: 8,043,892,736 || trainable%: 0.1694%
```

4. Trade-offs & PitfallsQuality vs. Quantity: The LIMA (Less Is More for Alignment) paper demonstrated that 1,000 well-written examples (SFT) outperform 50,000 mediocre examples. Invest in Data Curation.DPO vs RLHF: Today, PPO (classic RLHF) is rarely used due to its instability. DPO (Direct Preference Optimization) allows training directly on preferences (A is better than B) with the same efficiency.

## Phase 3: Production Inference Infrastructure (Inference Stack)

1. 30-Second OverviewThe biggest problem running an LLM in Production is not compute power (Compute Bound) but memory bandwidth (Memory Bound). The bottleneck is transferring the weights and the KV Cache from the GPU memory to the processor. The industry solution: vLLM with Continuous Batching and PagedAttention.
2. Mathematics / Memory Calculation (KV Cache)How much memory does the KV Cache require for a single token in LLaMA-3 8B (in FP16)?$$ \text{Cache Memory} = 2 \times \text{layers} \times \text{kv_heads} \times \text{head_dim} \times \text{bytes} $$$$ = 2 \times 32 \times 8 \times 128 \times 2 = 131,072 \text{ bytes per token} $$
For a context of 8,000 tokens for one user: $\sim 1 \text{ GB}$.
For a Batch of 64 users: $\sim 64 \text{ GB}$ is required just for context memory! (beyond the model weights).PagedAttention solves this by dividing memory into virtual contiguous blocks, just like an operating system, reducing memory waste from 50% to 4%.
3. Implementation Code (vLLM Engine)# To run this, install vLLM: pip install vllm
from vllm import LLM, SamplingParams

```python
# Initialize engine (handles Continuous Batching and PagedAttention automatically)
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    tensor_parallel_size=1, # Number of GPUs to span
    quantization="awq", # Use AWQ 4-bit quantization for production
    max_model_len=4096,
    gpu_memory_utilization=0.9
)

# Inference Parameters
sampling_params = SamplingParams(
    temperature=0.7,  # Balanced creativity
    top_p=0.9,        # Nucleus sampling
    max_tokens=512,
    presence_penalty=1.1 # Reduce repetition
)

prompts = [
    "Explain the architecture of a transformer.",
    "Write a python script for binary search."
]

# Generate (Requests are processed asynchronously and batched dynamically)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

4. Benchmarks & MetricsTTFT (Time To First Token): Critical for user experience (Streaming).TPS (Tokens Per Second): A measure of the system's Throughput. vLLM improves Throughput by 10x to 24x compared to standard HuggingFace Transformers under load.

## Phase 4: Agent Systems and RAG (Agentic Systems)

1. 30-Second OverviewA "naked" language model is just a statistical engine. To generate business value (Enterprise AI), we wrap it in RAG systems (to gain organizational knowledge it wasn't trained on) and Agentic Workflow patterns (allowing it to use tools, plan, and self-correct).
2. Mathematics: Vector Search in RAGFinding the most relevant document for a given query is typically done using Cosine Similarity between the query vector $q$ and the document vector $d$:$$ \text{sim}(q, d) = \frac{q \cdot d}{|q| |d|} $$
Where the dimensions usually range from 768 to 3072 (depending on the Embedding model).
3. Implementation Code (LangGraph Agent with Tools)Building a ReAct (Reasoning and Acting) based agent that uses LangGraph to create a Stateful loop.

```python
import json
import time
from typing import Annotated, Sequence
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import ToolMessage, HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

# ==========================================
# 1. Define Tools & Schemas (Strong Typing)
# ==========================================
# In production, docstrings and Pydantic schemas are critical: they are injected 
# directly into the LLM's prompt to teach it exactly how and when to use the tool.
# Without explicit schemas, models frequently hallucinate parameter names or formats.

class WeatherInput(BaseModel):
    location: str = Field(
        description="The city name and country/state, e.g., 'San Francisco, CA' or 'Tokyo, Japan'."
    )

@tool("get_weather", args_schema=WeatherInput)
def get_weather(location: str) -> str:
    """
    Returns the current weather conditions for a given location.
    Use this tool whenever the user asks for weather conditions, temperature, or forecasts.
    """
    # In a real environment, this would execute an HTTP request to an API (e.g., OpenWeatherMap).
    # We simulate input validation, which is a crucial guardrail. Malformed inputs 
    # from LLMs are incredibly common; always validate and sanitize before execution.
    if not location or len(location) < 2:
        return json.dumps({"error": "Please provide a valid, specific city name."})
    
    # Simulating network latency for observability trace demonstration
    time.sleep(0.4)
    
    # Returning structured data is a best practice for reliable tool utilization.
    # It allows the LLM to parse the exact fields it needs for the final answer 
    # without trying to regex or guess formatting from a raw string.
    return json.dumps({
        "location": location,
        "temperature": "25°C",
        "condition": "Sunny",
        "humidity": "60%",
        "wind_speed": "12 km/h",
        "source": "MockWeatherAPI"
    })

class TimeInput(BaseModel):
    timezone: str = Field(
        default="UTC", 
        description="The target timezone, e.g., 'America/New_York' or 'Europe/London'."
    )

@tool("get_current_time", args_schema=TimeInput)
def get_current_time(timezone: str = "UTC") -> str:
    """
    Returns the current local time and date for a specified timezone.
    Use this when the user asks about the time or needs temporal context for other data.
    """
    return json.dumps({
        "timezone": timezone, 
        "time": "14:00", 
        "date": "2024-10-24"
    })

tools = [get_weather, get_current_time]

# Create a lookup dictionary for robust, dynamic tool execution.
# This dispatcher pattern avoids massive if/else blocks and scales easily 
# to hundreds of enterprise tools (e.g., SQL query engines, CRM lookups).
tools_by_name = {tool.name: tool for tool in tools}

# ==========================================
# 2. Setup Model with Tools
# ==========================================
# Temperature 0 is highly preferred for agentic routing. It minimizes hallucinated 
# tool calls, ensures deterministic JSON argument generation, and prevents the 
# agent from getting overly "creative" with its execution paths.
llm = ChatOpenAI(model="gpt-4o", temperature=0, request_timeout=30)
llm_with_tools = llm.bind_tools(tools)

# ==========================================
# 3. Define State (Agent Memory)
# ==========================================
# The State defines the memory (context window) of our agent graph.
# In advanced production setups, this might also include metadata tracking 
# variables like 'user_id', 'auth_token', or 'recursion_depth'.
class State(TypedDict):
    # 'add_messages' is a reducer function. It ensures that returning new messages 
    # from any node strictly appends them to the existing list, rather than overwriting.
    # This preserves the full Thought -> Action -> Observation history.
    messages: Annotated[list, add_messages]

# ==========================================
# 4. Define Graph Nodes
# ==========================================
def chatbot(state: State):
    # The agent inspects the conversation history and decides on the next action (or final answer).
    # We wrap the invocation to catch potential API rate limits or connection errors gracefully.
    try:
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}
    except Exception as e:
        # Fallback mechanism if the LLM API fails, preventing a total crash of the pipeline.
        return {"messages": [AIMessage(content=f"System Error during generation: {e}")]}

def tool_executor(state: State):
    last_message = state["messages"][-1]
    tool_calls = last_message.tool_calls
    
    # Execute tools dynamically based on the LLM's explicit requests
    results = []
    for action in tool_calls:
        tool_name = action['name']
        tool_args = action['args']
        
        # Observability: In MLOps pipelines, we trace tool execution time and arguments
        # This data is usually sent to Datadog, LangSmith, or a similar telemetry backend.
        print(f"  [System Log] Dispatched tool: '{tool_name}' | Args: {tool_args}")
        start_time = time.time()
        
        if tool_name in tools_by_name:
            tool_instance = tools_by_name[tool_name]
            try:
                # Production agents must gracefully catch and pass execution exceptions
                # back to the LLM so the agent can naturally attempt a recovery,
                # self-correct its parameters, or politely inform the user of the failure.
                raw_result = tool_instance.invoke(tool_args)
                
                # Context Window Safeguard: Truncate massive tool outputs (like large SQL dumps)
                # to prevent exceeding the model's maximum context length.
                result = str(raw_result)[:4000] 
            except Exception as e:
                # Capture stack traces internally, but return a clean error to the LLM context
                result = f"Error executing tool: {str(e)}. Please try again with different parameters."
        else:
            result = f"Error: Tool '{tool_name}' not found in registry."
            
        execution_time = round(time.time() - start_time, 2)
        print(f"  [System Log] Tool '{tool_name}' completed in {execution_time}s")
        
        # Append the tool's output back into the message list with the required tool_call_id.
        # This acts as the "Observation" step in the ReAct framework.
        results.append(ToolMessage(content=str(result), tool_call_id=action['id']))
        
    return {"messages": results}

def route_condition(state: State):
    """
    Decide the next routing step dynamically.
    If the LLM returned tool calls (Action), route to the 'tools' node to get Observations.
    Otherwise, the LLM has provided its final synthesized text response, so route to END.
    """
    last_message = state["messages"][-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    return END

# ==========================================
# 5. Build Graph
# ==========================================
# This creates a robust state machine where nodes are functional blocks and edges are transitions.
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_executor)

graph_builder.set_entry_point("chatbot")
graph_builder.add_conditional_edges("chatbot", route_condition)

# Loop back to the chatbot so it can synthesize the final answer using the tool's findings.
# This loop continues until the LLM decides it has enough info to answer the user.
graph_builder.add_edge("tools", "chatbot") 

# Compile the graph into a runnable application structure
app = graph_builder.compile()

# ==========================================
# 6. Run the Agent Loop
# ==========================================
# We prime the agent with a System Message to enforce its persona and rules, 
# followed by the User query.
sys_msg = SystemMessage(
    content="You are a helpful, factual AI assistant. Always use your available tools "
            "to verify real-time facts before answering. Do not guess."
)
usr_msg = HumanMessage(content="What is the weather in Tel Aviv? Also, what time is it there?")

input_state = {"messages": [sys_msg, usr_msg]}

print("====================================")
print("Starting agentic loop execution...")
print("====================================\n")

# app.stream yields the state changes continuously as the agent navigates the graph
for event in app.stream(input_state):
    for node, values in event.items():
        print(f"--- Graph Transition: Leaving '{node}' Node ---")
        last_msg = values["messages"][-1]
        
        # Display either the standard synthesized text response or note the dispatched tools
        if last_msg.content:
            print(f"Agent Reply: {last_msg.content}")
        if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
            print(f"Agent Action: Decided to invoke -> {[tc['name'] for tc in last_msg.tool_calls]}")
        print("-" * 36)
```

4. Trade-offs & PitfallsNaive RAG vs. Advanced RAG: Standard RAG misses a lot of context. In Production, it is mandatory to use Reranking (a Cross-Encoder model that re-ranks the results from the Vector DB) to jump the accuracy from 70% to 90%+.Tool Execution: The main problem with Agents is hallucinations in tool parameters (passing Tel-Aviv instead of Tel Aviv if the API is sensitive). Using strict parameterization (Structured Output / JSON Mode) is mandatory.Paper to Read: ReAct: Synergizing Reasoning and Acting in Language Models (Yao et al., 2022).
