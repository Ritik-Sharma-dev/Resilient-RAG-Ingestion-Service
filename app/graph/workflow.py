from langgraph.graph import END, StateGraph

from app.graph.nodes import (
    GraphDependencies,
    make_format_node,
    make_generate_node,
    make_retrieve_node,
)
from app.graph.state import GraphState
from app.telemetry.langfuse_client import log_step, observe


def build_workflow(deps: GraphDependencies):
    graph = StateGraph(GraphState)

    graph.add_node("retrieve", make_retrieve_node(deps))
    graph.add_node("generate", make_generate_node(deps))
    graph.add_node("format", make_format_node(deps))

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "format")
    graph.add_edge("format", END)

    return graph.compile()


@observe(name="query")
def run_query_pipeline(deps: GraphDependencies, question: str, top_k: int | None) -> dict:
    """Parent trace for a /query request. Decorated so retrieve/generate/format
    (each @observe()-decorated in nodes.py) nest under this trace instead of
    each becoming its own top-level trace."""
    workflow = build_workflow(deps)
    result = workflow.invoke({"question": question, "top_k": top_k})

    log_step(
        name="query",
        metadata={
            "question": question,
            "source_count": len(result["structured_output"]["sources"]),
            "confidence": result["structured_output"]["confidence"],
        },
    )
    return result
