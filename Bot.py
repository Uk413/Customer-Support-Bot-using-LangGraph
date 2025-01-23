from typing import Dict, TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from dotenv import load_dotenv
import os

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv('GOOGLE_API_KEY')

class State(TypedDict):
    query: str
    category: str
    sentiment: str
    response: str

def categorize(state: State) -> State:
    """Categorize the customer query into Technical, Billing, or General."""
    prompt = ChatPromptTemplate.from_template(
        "Categorize the following customer query into one of these categories: "
        "Technical, Billing, General. Query: {query}"
    )
    chain = prompt | ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    category = chain.invoke({"query": state["query"]}).content.strip()
    return {"category": category}

def analyze_sentiment(state: State) -> State:
    """Analyze the sentiment of the customer query as Positive, Neutral, or Negative."""
    prompt = ChatPromptTemplate.from_template(
        "Analyze the sentiment of the following customer query. "
        "Respond with either 'Positive', 'Neutral', or 'Negative'. Query: {query}"
    )
    chain = prompt | ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    sentiment = chain.invoke({"query": state["query"]}).content.strip()
    return {"sentiment": sentiment}

def technical_query(state: State) -> State:
    """Provide a technical support response to the query."""
    prompt = ChatPromptTemplate.from_template(
        "Provide a technical support response to the following query: {query}"
    )
    chain = prompt | ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    response = chain.invoke({"query": state["query"]}).content.strip()
    return {"response": response}

def billing_query(state: State) -> State:
    """Provide a billing support response to the query."""
    prompt = ChatPromptTemplate.from_template(
        "Provide a billing support response to the following query: {query}"
    )
    chain = prompt | ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    response = chain.invoke({"query": state["query"]}).content.strip()
    return {"response": response}

def general(state: State) -> State:
    """Provide a general support response to the query."""
    prompt = ChatPromptTemplate.from_template(
        "Provide a general support response to the following query: {query}"
    )
    chain = prompt | ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    response = chain.invoke({"query": state["query"]}).content
    return {"response": response}

def search_web(state: State) -> State:
    """Search the web for a solution to the customer's query and provide a summarized response."""
    search_tool = DuckDuckGoSearchRun()
    search_results = search_tool.run(state["query"])
    if search_results:
        prompt = ChatPromptTemplate.from_template(
            "Summarize the following web search results into a concise and helpful response: {results}"
        )
        chain = prompt | ChatGoogleGenerativeAI(model="gemini-1.5-flash")
        summarized_response = chain.invoke({"results": search_results}).content.strip()
        return {"response": summarized_response}
    else:
        return {"response": "Can't find a solution to your query. Please mail your issue at assistance@xyz.com for further assistance."}

def route_query(state: State) -> str:
    """Route the query based on its sentiment and category."""
    if state["sentiment"] == "Negative":
        return "search_web"
    elif state["category"] == "Technical":
        return "technical_query"
    elif state["category"] == "Billing":
        return "billing_query"
    else:
        return "general"

workflow = StateGraph(State)
workflow.add_node("categorize", categorize)
workflow.add_node("analyze_sentiment", analyze_sentiment)
workflow.add_node("technical_query", technical_query)
workflow.add_node("billing_query", billing_query)
workflow.add_node("general", general)
workflow.add_node("search_web", search_web)
workflow.add_edge("categorize", "analyze_sentiment")
workflow.add_conditional_edges(
    "analyze_sentiment",
    route_query,
    {
        "technical_query": "technical_query",
        "billing_query": "billing_query",
        "general": "general",
        "search_web": "search_web"
    }
)
workflow.add_edge("technical_query", END)
workflow.add_edge("billing_query", END)
workflow.add_edge("general", END)
workflow.add_edge("search_web", END)
workflow.set_entry_point("categorize")
app = workflow.compile()

def run_customer_support(query: str) -> Dict[str, str]:
    """Process a customer query through the LangGraph workflow."""
    results = app.invoke({"query": query})
    return {
        "category": results.get("category", "Unknown"),
        "sentiment": results.get("sentiment", "Unknown"),
        "response": results.get("response", "No response generated")
    }

def main():
    """Command-line interface for the chatbot."""
    print("Hi! I'm your AI Assistant. How can I help you today?")
    print("Type 'exit' to quit the chatbot.")
    
    while True:
        user_query = input("\nYou: ").strip()
        if user_query.lower() == 'exit':
            print("Goodbye!")
            break
        try:
            result = run_customer_support(user_query)
            print(f"\nCategory: {result['category']}")
            print(f"Sentiment: {result['sentiment']}")
            print(f"Response: {result['response']}")
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
