# Tool Calling with Ollama & Llama 3.2

This document explains how to implement tool calling with Ollama running Llama 3.2, focusing specifically on how to pass tool results back to the model for final response construction.

## Overview of the Tool Calling Flow

1. **Initial Request**: Send user query with tool definitions
2. **Tool Call Detection**: Model may generate tool calls in response
3. **Tool Execution**: Execute the requested tools externally
4. **Result Passing**: Pass results back to model using proper format
5. **Final Response**: Model generates a final response incorporating tool results

## Key Components for Passing Tool Results

### 1. Tool Response Format

When passing tool results back to Llama 3.2, use the following message structure:

```json
{
  "role": "tool",
  "tool_call_id": "the-id-from-the-tool-call",
  "name": "tool_name",
  "content": "JSON-stringified-result"
}
```

### 2. Simplified Conversation Structure

For a more direct approach, you can omit the initial assistant response and simply pass the tool results to the model:

```json
[
  {"role": "user", "content": "original user query\n\nPlease use the following tool results in the construction of your response:"},
  {
    "role": "tool", 
    "tool_call_id": "call_123",
    "name": "get_weather",
    "content": "{\"location\":\"San Francisco\",\"temperature\":72,\"unit\":\"fahrenheit\",\"forecast\":[\"sunny\",\"windy\"],\"humidity\":45}"
  }
]
```

This approach is more efficient as it:
1. Skips the intermediate assistant response
2. Provides clear instructions to the model about using the tool results
3. Maintains the essential context without redundant information

## Implementation Details

### Error Handling

1. Always stringify the JSON result in the `content` field
2. Include the exact `tool_call_id` from the original call
3. Match the `name` field exactly to maintain context

### Multiple Tool Calls

When handling multiple tool calls with the simplified approach:
1. Pass all tool results as separate messages after the user instruction
2. Maintain the order of tool calls when possible
3. Example with multiple tools:

```json
[
  {"role": "user", "content": "query\n\nPlease use the following tool results in the construction of your response:"},
  {"role": "tool", "tool_call_id": "call_1", "name": "tool1", "content": "{}"},
  {"role": "tool", "tool_call_id": "call_2", "name": "tool2", "content": "{}"}
]
```

## Best Practices

1. **Validation**: Always validate tool results before passing them back
2. **Formatting**: Ensure proper JSON serialization of results
3. **Error States**: Pass error information in the result for failed tool calls
4. **Timeouts**: Handle tool execution timeouts gracefully
5. **Context Length**: Be mindful of context window limitations when passing large results

## Common Issues

1. **Incorrect IDs**: Mismatch between `tool_call_id` in request and response
2. **JSON Formatting**: Improperly escaped or malformed JSON strings
3. **Missing Fields**: Omitting required fields in the tool response
4. **Order Sensitivity**: Some models perform better when tool results maintain original call order

## Testing Considerations

1. Test with both successful and error cases
2. Verify model handles multiple sequential tool calls properly
3. Check model behavior with missing or partial results
