from typing import Annotated
import json

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_aws.chat_models import ChatBedrock
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage, SystemMessage

class State(TypedDict):

    # 更新された際、上書きじゃなくてmessageにadd_messagesを適用する
    messages: Annotated[list, add_messages]

model = ChatBedrock(model="amazon.nova-lite-v1:0", region="us-east-1")

@tool
def get_weather(location: str):
    """Call to get the weather from a specific location."""
    # This is a placeholder for the actual implementation
    # Don't let the LLM know this though 😊
    if any([city in location.lower() for city in ["sf", "san francisco"]]):
        return "It's sunny in San Francisco"
    else:
        return f"I am not sure what the weather is in {location}"
    
tools = [get_weather]

model = model.bind_tools(tools)


tools_by_name = {tool.name: tool for tool in tools}


# Define our tool node
def tool_node(state: State):
    outputs = []
    for tool_call in state["messages"][-1].tool_calls:
        tool_result = tools_by_name[tool_call["name"]].invoke(tool_call["args"])
        outputs.append(
            ToolMessage(
                content=json.dumps(tool_result),
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
            )
        )
    return {"messages": outputs}


def invoke_model(state: State):
    system_prompt = SystemMessage(
        "You are a helpful AI assistant, answer in japanese"
    )
    response = model.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}

def should_call_tool(state: State):
    if not state["messages"][-1].tool_calls:
        return "end"
    else:
        return "continue"


workflow = StateGraph(State)

workflow.add_node("agent", invoke_model)
workflow.add_node("tool_node", tool_node)

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    should_call_tool,
    {
        "continue": "tool_node",
        "end": END,
    },
)

workflow.add_edge("tool_node", "agent")

graph = workflow.compile()

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()


inputs = {"messages": ["sfの天気はどうですか？"]}
print_stream(graph.stream(inputs, stream_mode="values"))