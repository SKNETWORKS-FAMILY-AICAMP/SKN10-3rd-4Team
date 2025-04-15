import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from langgraph.graph import StateGraph, START, END
from PIL import Image
import tempfile
import matplotlib
import platform
import matplotlib.font_manager as fm

# 한글 폰트 설정
if platform.system() == 'Windows':
    # 나눔고딕 폰트 설정
    font_path = 'C:/Windows/Fonts/NanumGothic.ttf'
    if os.path.exists(font_path):
        font_prop = fm.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = [font_prop.get_name()]
        print(f"한글 폰트 설정 완료: NanumGothic")
    else:
        print(f"경고: {font_path} 폰트 파일이 존재하지 않습니다.")
else:
    # 다른 OS에서는 기본 sans-serif 사용
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

# 마이너스 기호 깨짐 방지
matplotlib.rcParams['axes.unicode_minus'] = False

def visualize_langgraph_workflow():
    """
    LangGraph 워크플로우를 시각화하는 함수
    
    Returns:
        str: 저장된 이미지 파일 경로
    """
    # NetworkX 그래프 생성
    G = nx.DiGraph()
    
    # 노드 추가 (영문 라벨 사용) - 줄바꿈 처리
    G.add_node(START, label="START")
    G.add_node("classify", label="Classify Question")
    G.add_node("route_by_type", label="Route by Type", shape="diamond")
    G.add_node("retrieve", label="Retrieve Documents")
    G.add_node("generate_academic", label="Generate\nAcademic\nResponse")  # 줄바꿈 추가
    G.add_node("generate_counseling", label="Generate\nCounseling\nResponse")  # 줄바꿈 추가
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
    
    # 노드 크기 증가 (4500에서 5500으로)
    node_size = 5500
    
    for node in G.nodes():
        nx.draw_networkx_nodes(
            G, pos, 
            nodelist=[node], 
            node_size=node_size, 
            node_color=node_colors[node],
            edgecolors='black'
        )
    
    # 엣지 그리기 - 노드 크기 매개변수도 업데이트
    nx.draw_networkx_edges(G, pos, width=2, arrowsize=20, arrowstyle='->', node_size=node_size)
    
    # 노드 라벨 그리기 - 폰트 크기 조정
    node_labels = {node: G.nodes[node]["label"] for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10, font_weight="bold")
    
    # 엣지 라벨 그리기
    edge_labels = {(u, v): d["label"] for u, v, d in G.edges(data=True) if d["label"]}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)
    
    # 그래프 제목과 설정
    plt.title("Depression Chatbot LangGraph Workflow", fontsize=16, pad=20)
    plt.axis("off")
    plt.tight_layout()
    
    # 더 여유 있는 레이아웃
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    
    # 이미지 저장
    # '프로젝트 루트/visualization' 디렉토리에 저장
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    output_dir = os.path.join(project_root, "visualization")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "langgraph_workflow.png")
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
    
    # 노드 추가 (영문 라벨 사용) - 줄바꿈 처리
    G.add_node(START, label="START")
    G.add_node("classify", label="Question Classification")
    G.add_node("retrieve", label="Document Retrieval")
    G.add_node("generate_academic", label="Academic\nResponse")  # 줄바꿈 추가
    G.add_node("generate_counseling", label="Counseling\nResponse")  # 줄바꿈 추가
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
    
    # 노드 크기 증가 (4000에서 5000으로)
    node_size = 5000
    
    for node in G.nodes():
        nx.draw_networkx_nodes(
            G, pos, 
            nodelist=[node], 
            node_size=node_size, 
            node_color=node_colors[node],
            edgecolors='black'
        )
    
    # 엣지 그리기 - 노드 크기 매개변수도 업데이트
    nx.draw_networkx_edges(G, pos, width=2, arrowsize=20, arrowstyle='->', node_size=node_size)
    
    # 노드 라벨 그리기 - 폰트 크기 조정
    node_labels = {node: G.nodes[node]["label"] for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=11, font_weight="bold")
    
    # 엣지 라벨 그리기
    edge_labels = {(u, v): d["label"] for u, v, d in G.edges(data=True) if d["label"]}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)
    
    # 그래프 제목과 설정
    plt.title("Depression Chatbot LangGraph Workflow", fontsize=16, pad=20)
    plt.axis("off")
    plt.tight_layout()
    
    # 더 여유 있는 레이아웃
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    
    # 이미지 저장
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    output_dir = os.path.join(project_root, "visualization")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "simple_langgraph_workflow.png")
    plt.savefig(output_path, format="png", dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"Simple workflow image saved: {output_path}")
    return output_path

def visualize_current_workflow():
    """
    현재 사용 중인 LangGraph 워크플로우를 시각화하는 함수
    
    Returns:
        str: 저장된 이미지 파일 경로
    """
    # NetworkX 그래프 생성
    G = nx.DiGraph()
    
    # 노드 추가 (한글 설명 포함) - 길이가 긴 라벨은 줄바꿈 처리
    G.add_node(START, label="START")
    G.add_node("classify", label="질문 분류")
    G.add_node("retrieve", label="논문 검색")
    G.add_node("route_academic_search", label="유사도 판단", shape="diamond")
    G.add_node("tavily_search", label="웹 검색")
    G.add_node("generate_academic", label="학술\n응답 생성")  # 줄바꿈 추가
    G.add_node("generate_counseling", label="상담\n응답 생성")  # 줄바꿈 추가
    G.add_node("retrieve_youtube", label="유튜브 검색")
    G.add_node(END, label="END")
    
    # 엣지 추가
    G.add_edge(START, "classify", label="")
    G.add_edge("classify", "retrieve", label="학술 질문")
    G.add_edge("classify", "retrieve_youtube", label="상담 질문")
    G.add_edge("retrieve", "route_academic_search", label="")
    G.add_edge("route_academic_search", "tavily_search", label="유사도 낮음")
    G.add_edge("route_academic_search", "generate_academic", label="유사도 높음")
    G.add_edge("tavily_search", "generate_academic", label="")
    G.add_edge("retrieve_youtube", "generate_counseling", label="")
    G.add_edge("generate_academic", END, label="")
    G.add_edge("generate_counseling", END, label="")
    
    # 그래프 레이아웃 설정
    pos = {
        START: np.array([0, 0]),
        "classify": np.array([1, 0]),
        "retrieve": np.array([2, 0.5]),
        "retrieve_youtube": np.array([2, -0.5]),
        "route_academic_search": np.array([3, 0.5]),
        "tavily_search": np.array([4, 1]),
        "generate_academic": np.array([5, 0.5]),
        "generate_counseling": np.array([3, -0.5]),
        END: np.array([6, 0])
    }
    
    # 그래프 시각화
    plt.figure(figsize=(16, 10))
    
    # 노드 그리기
    node_colors = {
        START: "lightgreen", 
        "classify": "lightyellow",
        "retrieve": "lightblue",
        "route_academic_search": "lightgray",
        "tavily_search": "lightpink",
        "generate_academic": "lightblue",
        "generate_counseling": "lightpink",
        "retrieve_youtube": "lightblue",
        END: "lightcoral"
    }
    
    # 노드 크기 증가 (4000에서 5000으로)
    node_size = 5000
    
    for node in G.nodes():
        nx.draw_networkx_nodes(
            G, pos, 
            nodelist=[node], 
            node_size=node_size, 
            node_color=node_colors[node],
            edgecolors='black'
        )
    
    # 엣지 그리기 - 노드 크기 매개변수도 업데이트
    nx.draw_networkx_edges(G, pos, width=2, arrowsize=20, arrowstyle='->', node_size=node_size)
    
    # 노드 라벨 그리기 - 폰트 크기 조정
    node_labels = {node: G.nodes[node]["label"] for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=11, font_weight="bold")
    
    # 엣지 라벨 그리기
    edge_labels = {(u, v): d["label"] for u, v, d in G.edges(data=True) if d["label"]}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)
    
    # 그래프 제목과 설정
    plt.title("우울증 챗봇 LangGraph 워크플로우", fontsize=16, pad=20)
    plt.axis("off")
    plt.tight_layout()
    
    # 더 여유 있는 레이아웃
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    
    # 이미지 저장
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    output_dir = os.path.join(project_root, "visualization")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "current_workflow.png")
    plt.savefig(output_path, format="png", dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"현재 워크플로우 이미지 저장: {output_path}")
    return output_path

if __name__ == "__main__":
    # 복잡한 워크플로우 시각화
    image_path = visualize_langgraph_workflow()
    Image.open(image_path).show()
    
    # 단순한 워크플로우 시각화
    simple_image_path = visualize_simple_langgraph_workflow()
    Image.open(simple_image_path).show()
    
    # 현재 사용 중인 워크플로우 시각화
    current_image_path = visualize_current_workflow()
    Image.open(current_image_path).show() 