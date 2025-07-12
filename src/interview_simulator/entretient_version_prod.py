import os
import sys
import json
from typing import Dict, List, Any, Annotated
from typing_extensions import TypedDict

from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode 
from langchain_openai import ChatOpenAI

from src.config import read_system_prompt, format_cv
from src.crew.crew_pool import interview_analyser 


class State(TypedDict):
    messages: Annotated[list, add_messages]

class InterviewProcessor:
    def __init__(self, cv_document: Dict[str, Any], job_offer: Dict[str, Any], conversation_history: List[Dict[str, Any]]):
        if not cv_document or 'candidat' not in cv_document:
            raise ValueError("Document CV invalide fourni.")
        if not job_offer:
            raise ValueError("Données de l'offre d'emploi non fournies.")

        self.job_offer = job_offer
        self.cv_data = cv_document['candidat']
        self.conversation_history = conversation_history
        self.tools = [interview_analyser]
        self.llm = self._get_llm()
        self.llm_with_tools = self.llm.bind_tools(self.tools)

        self.system_prompt_template = self._load_prompt_template()
        self.graph = self._build_graph()

    def _get_llm(self) -> ChatOpenAI:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        return ChatOpenAI(
        temperature=0.6, 
        model_name="gpt-4o-mini", 
        api_key=openai_api_key
    )

    def _load_prompt_template(self) -> str:
        return read_system_prompt('prompts/rag_prompt.txt')

    def _chatbot_node(self, state: State) -> dict:
        if state["messages"] and isinstance(state["messages"][-1], ToolMessage):
            tool_message = state["messages"][-1]
            return {"messages": [AIMessage(content=tool_message.content)]}
        messages = state["messages"]
        formatted_cv_str = format_cv(self.cv_data)
        system_prompt = self.system_prompt_template.format(
            entreprise=self.job_offer.get('entreprise', 'notre entreprise'),
            poste=self.job_offer.get('poste', 'ce poste'),
            description=self.job_offer.get('description', 'la description du poste'),
            cv=formatted_cv_str
        )
        llm_messages = [SystemMessage(content=system_prompt)] + messages
        response = self.llm_with_tools.invoke(llm_messages)
        return {"messages": [response]}

    def _route_after_chatbot(self, state: State) -> str:
        last_message = state["messages"][-1]
        if last_message.tool_calls:
            return "call_tool"
        return END

    def _build_graph(self) -> any:
        graph_builder = StateGraph(State)
        
        graph_builder.add_node("chatbot", self._chatbot_node)
        graph_builder.add_node("call_tool", ToolNode(self.tools))        
        graph_builder.add_edge(START, "chatbot")        
        graph_builder.add_conditional_edges(
            "chatbot",
            self._route_after_chatbot,
            {
                "call_tool": "call_tool", 
                END: END                  
            }
        )
        graph_builder.add_edge("call_tool", "chatbot")
        return graph_builder.compile()

    def run(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        initial_state = self.conversation_history + messages
        return self.graph.invoke({"messages": initial_state})