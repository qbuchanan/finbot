import json
import requests
from typing import Dict, List, Any, Optional

# Define our tool schema
TOOLS = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g., San Francisco, CA"
                }
            },
            "required": ["location"]
        }
    }
]

# Mock weather tool implementation
def get_weather(location: str) -> Dict[str, Any]:
    """Simulate getting weather data for a location"""
    # In a real app, you'd call a weather API here
    return {
        "location": location,
        "temperature": 72,
        "unit": "fahrenheit",
        "forecast": ["sunny", "windy"],
        "humidity": 45
    }

def execute_tool_call(tool_call: Dict[str, Any]) -> Dict[str, Any]:
    """Execute the requested tool and return results"""
    tool_name = tool_call.get("name")
    arguments = json.loads(tool_call.get("arguments", "{}"))
    
    if tool_name == "get_weather":
        result = get_weather(arguments.get("location"))
        return {
            "tool_call_id": tool_call.get("id"),
            "name": tool_name,
            "result": json.dumps(result)
        }
    else:
        return {
            "tool_call_id": tool_call.get("id"),
            "name": tool_name,
            "result": json.dumps({"error": "Unknown tool"})
        }

def chat_with_tools(
    prompt: str,
    model: str = "llama3.2:70b",
    tools: Optional[List[Dict[str, Any]]] = None,
    temperature: float = 0.7,
) -> Dict[str, Any]:
    """Send a prompt to Ollama with tool definitions and handle tool calls"""
    
    base_url = "http://localhost:11434/api/chat"
    
    # Initial request with tools definition
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "temperature": temperature,
    }
    
    if tools:
        payload["tools"] = tools
    
    # Send initial request
    response = requests.post(base_url, json=payload)
    response_data = response.json()
    
    # Check if there are tool calls to process
    assistant_message = response_data.get("message", {})
    tool_calls = assistant_message.get("tool_calls", [])
    
    if not tool_calls:
        # No tool calls, return the response as is
        return response_data
    
    # Process tool calls and gather results
    tool_results = []
    for tool_call in tool_calls:
        result = execute_tool_call(tool_call)
        tool_results.append(result)
    
    # Update conversation with tool results and get final response
    new_messages = [
        {"role": "user", "content": prompt + "\n\nPlease use the following tool results in the construction of your response:"},
        {
            "role": "tool", 
            "tool_call_id": tool_results[0]["tool_call_id"],
            "name": tool_results[0]["name"],
            "content": tool_results[0]["result"]
        }
    ]
    
    final_payload = {
        "model": model,
        "messages": new_messages,
        "stream": False,
        "temperature": temperature,
    }
    
    # Send follow-up request with tool results
    final_response = requests.post(base_url, json=final_payload)
    return final_response.json()

# Example usage
if __name__ == "__main__":
    user_query = "What's the weather like in San Francisco?"
    response = chat_with_tools(user_query, tools=TOOLS)
    
    # Extract the final response
    if "message" in response:
        print(f"Final response: {response['message']['content']}")
    else:
        print(f"Error or unexpected response: {response}")
