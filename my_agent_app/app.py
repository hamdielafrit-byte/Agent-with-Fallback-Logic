from flask import Flask, render_template, request, jsonify
from duckduckgo_search import DDGS 
import json
import math
import ollama

app = Flask(__name__)

# --- Advanced Memory Repositories ---
CHAT_HISTORY = []
LONG_TERM_FACTS = []  # Acts as our long-term memory graph/fact index

# --- Tools Definition ---
def calculator_tool(expression: str) -> str:
    allowed_chars = "0123456789+-*/(). sqrt"
    if not all(c in allowed_chars for c in expression):
        raise ValueError("Invalid characters in mathematical expression.")
    safe_expr = expression.replace("sqrt", "math.sqrt")
    result = eval(safe_expr, {"__builtins__": None, "math": math})
    return str(result)

def search_tool(query: str) -> str:
    if "404" in query.lower():
        raise RuntimeError("Search API returned a 404 Error: Service Unavailable.")
    try:
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(query, max_results=3)]
        if not results:
            return "No live web results found."
        context = ""
        for i, r in enumerate(results, 1):
            context += f"Source {i}: {r['title']} - {r['body']}\n"
        return context
    except Exception as e:
        raise RuntimeError(f"Live web search failed: {str(e)}")


# --- Agent Core with Memory Optimization ---
class SmartAgent:
    def __init__(self, model_name: str = "qwen2.5:3b"):
        self.model_name = model_name
        
    def _update_long_term_memory(self, history: list):
        """Condenses oldest messages into facts to keep VRAM usage low."""
        global LONG_TERM_FACTS
        # Only extract facts if history is growing long to preserve speed
        if len(history) <= 4:
            return
            
        # Extract the messages that are falling out of the sliding window
        old_exchange = history[:-4]
        text_to_summarize = ""
        for msg in old_exchange:
            text_to_summarize += f"{msg['role']}: {msg['content']}\n"
            
        system_instruction = (
            "You are a memory condensation system. Extract critical, permanent facts about the user "
            "(names, preferences, favorites, background info) from the text. Respond ONLY with a bulleted "
            "list of new facts, or reply 'None' if no critical personal facts are found."
        )
        
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": f"Existing Facts:\n{LONG_TERM_FACTS}\n\nNew Thread:\n{text_to_summarize}"}
                ],
                options={"temperature": 0.0, "num_predict": 100}
            )
            new_facts = response['message']['content'].strip()
            if "none" not in new_facts.lower():
                # Store the updated facts smoothly
                LONG_TERM_FACTS.append(new_facts)
        except Exception:
            pass # Fail gracefully in the background

    def _route_input(self, user_prompt: str, history: list) -> dict:
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
        for msg in history[-4:]: 
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": user_prompt})

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=messages,
                options={"temperature": 0.0, "num_predict": 40, "num_ctx": 2048}
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
        messages = []
        
        # Inject our consolidated long term facts into the system instruction
        facts_str = "\n".join(LONG_TERM_FACTS) if LONG_TERM_FACTS else "None yet."
        
        system_content = (
            "You are a friendly, helpful AI buddy. Chat naturally with the user.\n"
            "CRITICAL: You have access to the chat history and long-term knowledge graphs.\n"
            f"KNOWN PERMANENT FACTS ABOUT USER:\n{facts_str}\n\n"
            "If the user references something old, use the facts above to remember seamlessly!"
        )
        
        if context:
            system_content += f"\n\nUse this live internet context data to answer current info queries:\n{context}"
            
        messages.append({"role": "system", "content": system_content})
            
        # Recent sliding window to prevent high text stack overhead
        for msg in history[-4:]: 
            messages.append({"role": msg["role"], "content": msg["content"]})
            
        messages.append({"role": "user", "content": user_prompt})
        
        response = ollama.chat(
            model=self.model_name, 
            messages=messages,
            options={"num_ctx": 4096}
        )
        return response['message']['content']

    def handle_query(self, user_prompt: str, history: list) -> tuple[str, str]:
        # Update our condensed background facts right before routing
        self._update_long_term_memory(history)
        
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
                if len(search_data.strip()) < 10:
                    return "Fallback Route (Low Tool Confidence)", self._direct_llm_answer(user_prompt, history)
                
                return "Tool Execution (Search)", self._direct_llm_answer(user_prompt, history, context=search_data)
            except Exception as e:
                return "Fallback Route (Search Failed)", self._direct_llm_answer(user_prompt, history)

        return "Error", "Unknown state."

# Instantiate Agent
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