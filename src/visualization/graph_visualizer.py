import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from langgraph.graph import StateGraph, START, END
from PIL import Image
import tempfile
import matplotlib
import platform

# 기본 폰트 설정 - 한글 폰트 문제로 영문 폰트 사용
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

def visualize_langgraph_workflow():
    """
    LangGraph 워크플로우를 시각화하는 함수
    
    Returns:
        str: 저장된 이미지 파일 경로
    """
    # NetworkX 그래프 생성
    G = nx.DiGraph()
    
    # 노드 추가 (영문 라벨 사용)
    G.add_node(START, label="START")
    G.add_node("classify", label="Classify Question")
    G.add_node("route_by_type", label="Route by Type", shape="diamond")
    G.add_node("retrieve", label="Retrieve Documents")
    G.add_node("generate_academic", label="Generate Academic Response")
    G.add_node("generate_counseling", label="Generate Counseling Response") 
    G.add_node(END, label="END")
    
    # 엣지 추가 (영문 라벨 사용)
    G.add_edge(START, "classify", label="")
    G.add_edge("classify", "route_by_type", label="")
    G.add_edge("route_by_type", "retrieve", label="Academic")
    G.add_edge("route_by_type", "generate_counseling", label="Counseling")
    G.add_edge("retrieve", "generate_academic", label="")
    G.add_edge("generate_academic", END, label="")
    G.add_edge("generate_counseling", END, label="")
    
    # 그래프 레이아웃 설정 (더 복잡한 레이아웃)
    pos = {
        START: np.array([0, 0]),
        "classify": np.array([1, 0]),
        "route_by_type": np.array([2, 0]),
        "retrieve": np.array([3, 0.5]),
        "generate_academic": np.array([4, 0.5]),
        "generate_counseling": np.array([3, -0.5]),
        END: np.array([5, 0])
    }
    
    # 그래프 시각화
    plt.figure(figsize=(14, 8))
    
    # 노드 그리기
    node_colors = {
        START: "lightgreen", 
        "classify": "lightyellow",
        "route_by_type": "lightgray",
        "retrieve": "lightblue", 
        "generate_academic": "lightblue",
        "generate_counseling": "lightpink",
        END: "lightcoral"
    }
    
    for node in G.nodes():
        nx.draw_networkx_nodes(
            G, pos, 
            nodelist=[node], 
            node_size=3000, 
            node_color=node_colors[node],
            edgecolors='black'
        )
    
    # 엣지 그리기
    nx.draw_networkx_edges(G, pos, width=2, arrowsize=20, arrowstyle='->', node_size=3000)
    
    # 노드 라벨 그리기
    node_labels = {node: G.nodes[node]["label"] for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=14, font_weight="bold")
    
    # 엣지 라벨 그리기
    edge_labels = {(u, v): d["label"] for u, v, d in G.edges(data=True) if d["label"]}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=12)
    
    # 그래프 제목과 설정
    plt.title("Depression Chatbot LangGraph Workflow", fontsize=16, pad=20)
    plt.axis("off")
    plt.tight_layout()
    
    # 이미지 저장
    # 'visualization' 디렉토리가 없으면 생성
    os.makedirs("visualization", exist_ok=True)
    output_path = "visualization/langgraph_workflow.png"
    plt.savefig(output_path, format="png", dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"Graph image saved: {output_path}")
    return output_path

def visualize_simple_langgraph_workflow():
    """
    단순한 LangGraph 워크플로우를 시각화하는 함수
    
    Returns:
        str: 저장된 이미지 파일 경로
    """
    # NetworkX 그래프 생성
    G = nx.DiGraph()
    
    # 노드 추가 (영문 라벨 사용)
    G.add_node(START, label="START")
    G.add_node("classify", label="Question Classification")
    G.add_node("retrieve", label="Document Retrieval")
    G.add_node("generate_academic", label="Academic Response")
    G.add_node("generate_counseling", label="Counseling Response")
    G.add_node(END, label="END")
    
    # 엣지 추가 (영문 라벨 사용)
    G.add_edge(START, "classify", label="")
    G.add_edge("classify", "retrieve", label="Academic")
    G.add_edge("classify", "generate_counseling", label="Counseling")
    G.add_edge("retrieve", "generate_academic", label="")
    G.add_edge("generate_academic", END, label="")
    G.add_edge("generate_counseling", END, label="")
    
    # 그래프 레이아웃 설정
    pos = {
        START: np.array([0, 0]),
        "classify": np.array([1, 0]), 
        "retrieve": np.array([2, 0.5]),
        "generate_academic": np.array([3, 0.5]),
        "generate_counseling": np.array([2, -0.5]),
        END: np.array([4, 0])
    }
    
    # 그래프 시각화
    plt.figure(figsize=(12, 6))
    
    # 노드 그리기
    node_colors = {
        START: "lightgreen", 
        "classify": "lightyellow",
        "retrieve": "lightblue", 
        "generate_academic": "lightblue",
        "generate_counseling": "lightpink",
        END: "lightcoral"
    }
    
    for node in G.nodes():
        nx.draw_networkx_nodes(
            G, pos, 
            nodelist=[node], 
            node_size=3000, 
            node_color=node_colors[node],
            edgecolors='black'
        )
    
    # 엣지 그리기
    nx.draw_networkx_edges(G, pos, width=2, arrowsize=20, arrowstyle='->', node_size=3000)
    
    # 노드 라벨 그리기
    node_labels = {node: G.nodes[node]["label"] for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=12, font_weight="bold")
    
    # 엣지 라벨 그리기
    edge_labels = {(u, v): d["label"] for u, v, d in G.edges(data=True) if d["label"]}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)
    
    # 그래프 제목과 설정
    plt.title("Depression Chatbot LangGraph Workflow", fontsize=16, pad=20)
    plt.axis("off")
    plt.tight_layout()
    
    # 이미지 저장
    os.makedirs("visualization", exist_ok=True)
    output_path = "visualization/simple_langgraph_workflow.png"
    plt.savefig(output_path, format="png", dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"Simple workflow image saved: {output_path}")
    return output_path

if __name__ == "__main__":
    # 복잡한 워크플로우 시각화
    image_path = visualize_langgraph_workflow()
    Image.open(image_path).show()
    
    # 단순한 워크플로우 시각화
    simple_image_path = visualize_simple_langgraph_workflow()
    Image.open(simple_image_path).show() 