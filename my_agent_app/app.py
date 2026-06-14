from flask import Flask, render_template, request, jsonify
from duckduckgo_search import DDGS  # Real live web search engine
import json
import math
import ollama

app = Flask(__name__)

# --- In-Memory Chat History ---
CHAT_HISTORY = []

# --- Real Tools Definition ---
def calculator_tool(expression: str) -> str:
    allowed_chars = "0123456789+-*/(). sqrt"
    if not all(c in allowed_chars for c in expression):
        raise ValueError("Invalid characters in mathematical expression.")
    safe_expr = expression.replace("sqrt", "math.sqrt")
    result = eval(safe_expr, {"__builtins__": None, "math": math})
    return str(result)

def search_tool(query: str) -> str:
    """Connects your agent directly to the live internet via DuckDuckGo."""
    # Kept intact so you can still explicitly test your 404 fallback logic!
    if "404" in query.lower():
        raise RuntimeError("Search API returned a 404 Error: Service Unavailable.")
    
    try:
        # Fetch the top 3 live text results from the web
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(query, max_results=3)]
        
        if not results:
            return "No live web results found."
            
        # Format the live web snippets into a single text block
        context = ""
        for i, r in enumerate(results, 1):
            context += f"Source {i}: {r['title']} - {r['body']}\n"
        return context
        
    except Exception as e:
        # If internet is down or search rate-limits us, raise error to kick off the fallback route smoothly
        raise RuntimeError(f"Live web search failed: {str(e)}")


# --- Agent Core with Speed Optimizations & Memory ---
class SmartAgent:
    def __init__(self, model_name: str = "qwen2.5:3b"):
        self.model_name = model_name
        
    def _route_input(self, user_prompt: str, history: list) -> dict:
        """Routes input swiftly by capping token generation and context limits."""
        system_instruction = (
            "You are an AI router. Analyze the user prompt and the recent chat history to decide the best action.\n"
            "If the user is asking about current events, news, weather, or real-time web facts, route to 'search'.\n"
            "Respond ONLY with a raw JSON object matching this schema:\n"
            "{\n"
            '  "action": "calculator" | "search" | "direct_reasoning",\n'
            '  "tool_input": "the extracted query or math expression, or empty string"\n'
            "}"
        )
        
        messages = [{"role": "system", "content": system_instruction}]
        for msg in history[-4:]: # Only pass the last 4 messages to save context processing time
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": user_prompt})

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=messages,
                options={
                    "temperature": 0.0,
                    "num_predict": 40,   # Capped generation: stops immediately after JSON is output
                    "num_ctx": 2048,     # Small context window for faster token routing
                }
            )
            content = response['message']['content'].strip()
            if content.startswith("```json"):
                content = content.split("```json")[1].split("```")[0].strip()
            elif content.startswith("```"):
                content = content.split("```")[1].split("```")[0].strip()
            return json.loads(content)
        except Exception:
            return {"action": "direct_reasoning", "tool_input": ""}

    def _direct_llm_answer(self, user_prompt: str, history: list, context: str = "") -> str:
        """Generates an answer using a sliding history window and injected live context."""
        messages = []
        
        # ─── UPDATED SYSTEM PROMPTS TO FORCE MEMORY RECOGNITION ───
        if context:
            messages.append({
                "role": "system", 
                "content": (
                    "You are a friendly AI buddy. Answer the user's prompt using this live internet context data:\n"
                    f"{context}\n"
                    "Note: You HAVE access to the history below. Never say you don't remember."
                )
            })
        else:
            messages.append({
                "role": "system", 
                "content": (
                    "You are a friendly, helpful AI buddy. Chat naturally with the user.\n"
                    "CRITICAL: You HAVE full access to the recent chat history provided below. "
                    "If the user asks you what you talked about, what their name is, or what they said earlier, "
                    "look at the history logs below and answer accurately. Never claim you don't have memory."
                )
            })
            
        # Sliding history window to keep generation fast and lean
        for msg in history[-4:]: 
            messages.append({"role": msg["role"], "content": msg["content"]})
            
        messages.append({"role": "user", "content": user_prompt})
        
        response = ollama.chat(
            model=self.model_name, 
            messages=messages,
            options={
                "num_ctx": 4096  
            }
        )
        return response['message']['content']

    def handle_query(self, user_prompt: str, history: list) -> tuple[str, str]:
        decision = self._route_input(user_prompt, history)
        action = decision.get("action", "direct_reasoning")
        tool_input = decision.get("tool_input", "")
        
        if action == "direct_reasoning":
            return "Direct Reasoning Route", self._direct_llm_answer(user_prompt, history)
            
        elif action == "calculator":
            try:
                result = calculator_tool(tool_input)
                return "Tool Execution (Calculator)", f"Result: {result}"
            except Exception as e:
                return "Fallback Route (Calculator Failed)", self._direct_llm_answer(user_prompt, history)
                
        elif action == "search":
            try:
                search_data = search_tool(tool_input)
                # Confidence Fallback Check (If data returned is suspiciously empty)
                if len(search_data.strip()) < 10:
                    return "Fallback Route (Low Tool Confidence)", self._direct_llm_answer(user_prompt, history)
                
                return "Tool Execution (Search)", self._direct_llm_answer(user_prompt, history, context=search_data)
            except Exception as e:
                return "Fallback Route (Search Failed)", self._direct_llm_answer(user_prompt, history)

        return "Error", "Unknown state."

# Instantiate Agent (Using qwen2.5:3b as default since it is fast and efficient)
agent = SmartAgent(model_name="qwen2.5:3b")

# --- Flask Routes ---
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    global CHAT_HISTORY
    user_input = request.json.get("query", "")
    if not user_input:
        return jsonify({"route": "Error", "response": "Empty prompt."}), 400
    
    route, response = agent.handle_query(user_input, CHAT_HISTORY)
    
    CHAT_HISTORY.append({"role": "user", "content": user_input})
    CHAT_HISTORY.append({"role": "assistant", "content": response})
    
    return jsonify({"route": route, "response": response})

if __name__ == "__main__":
    app.run(debug=True, port=5000)