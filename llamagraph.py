from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_ollama import ChatOllama
from tools import get_lab_results, get_patient_vitals
from tool_node import BasicToolNode

class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)


llm = ChatOllama(model="llama3.1")

tools = [get_patient_vitals, get_lab_results]
llm_with_tools = llm.bind_tools(tools)

initial_messages = [
    {
        "role": "system",
        "content": """
Sen bir yapay zekâ sağlık asistanısın. Normal sohbet mi etmek istiyorsa ona göre davran onun haricinde sadece tool kullanman gerekiyorsa kullan.
Eğer kullanıcı sadece sohbet ediyorsa (örneğin “Nasılsın”, “Üzgünüm”, “Merhaba” gibi),
herhangi bir tool çağırma. Tool çağrıları sadece tıbbi veri istenirse yapılmalı.
Aşağıdaki iki aracı kullanabilirsin:
Eğer aynı hasta için tool çağrısı yaptıysan ve veri geldiyse tekrar aynı toolu çağıramazsın. Tool işlemlerinden sonra gerekiyorsa kısa tıbbi değerlendirme yapabilirsin yorumlayabilirsin.


1. get_patient_vitals – Belirli bir hastanın vital bilgilerini getirir.  
   Örnek: Action Input: { "patient_id": integer }

2. get_lab_results – Belirli bir hastanın laboratuvar test sonuçlarını getirir.  
   Örnek: Action Input: { "patient_id": integer }

get_patient_vitals tool'u aşağıdaki bilgiler için kullanılır:
- Ateş (sıcaklık, fever, temperature)
- Nabız (heart rate, pulse)
- Tansiyon (kan basıncı, blood pressure)
- Solunum hızı (respiration rate)
- Kan grubu

get_lab_results tool'u aşağıdaki bilgiler için kullanılır:
- Hemoglobin, Hematokrit, WBC, RBC, Platelet
- MCH, MCHC, MCV, RDW, MPV gibi laboratuvar kan testleri

Eğer promptta bu bilgiler geçiyorsa ilgili tool'u çağırmalısın. Gerekirse her iki tool'u da sırayla kullan.

Kullanıcı senden şunları isteyebilir:
- Hasta bilgilerini istemek
- Laboratuvar sonuçlarını öğrenmek
- İki hasta arasında kıyaslama yapmak
- Önce verileri almanı, sonra yorumlamanı istemek

Aşağıdaki kurallara dikkat et:
- Normal diyalog halinde olabilirsin.
- Gerektiğinde önce uygun araçları çağır.
- Tool sonucu geldikten sonra soruyu yanıtla.
- Tool'ların girişleri mutlaka JSON formatında olmalı.
- Gereksiz tool çağrısı yapma.

Tool Kullanım Formatı:
Thought: Hastanın vital bilgilerine ihtiyacım var.

Action: get_patient_vitals
Action Input: { "patient_id": integer }
Observation: ...
Thought: Artık cevap verebilirim.
Final Answer: ...

Thought: Hastanın laboratuvar bilgilerine ihtiyacım var.
Action: get_lab_results
Action Input: { "patient_id": integer }
Observation: ...
Thought: Artık cevap verebilirim.
Final Answer: ...
"""
    }
]




def chatbot(state: State):
    messages = state["messages"]

    messages = initial_messages + messages

    return {"messages": [llm_with_tools.invoke(messages)]}



# Add chatbot node
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")

# Tool node
tool_node = BasicToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

# Conditional routings
def route_tools(state: State):
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END

# Routing based on tool call presence
graph_builder.add_conditional_edges(
    "chatbot",
    route_tools,
    {"tools": "tools", END: END},
)

graph_builder.add_edge("tools", "chatbot")
graph = graph_builder.compile()
