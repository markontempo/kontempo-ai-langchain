# kontempo_ai_langchain.py
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langserve import add_routes
from fastapi import FastAPI
import uvicorn
from typing import List, Dict
import json

# Configuración
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Sistema de prompt
SYSTEM_PROMPT = """Eres un asistente especializado para usuarios del dashboard de merchants de Kontempo.

CONTEXTO:
- Kontempo es un SaaS que ayuda a merchants a manejar programas de crédito
- Tú ayudas a los usuarios del dashboard a entender y gestionar su programa de crédito

ESTRUCTURA DE DATOS:
- buyers[] = Los clientes/buyers del merchant que usan el crédito
- orders[] = Transacciones/órdenes hechas por los buyers
- payouts[] = Pagos recibidos por el merchant (sus ingresos)
- payment_links[] = Links de pago pendientes (pipeline de ventas)

REGLA PRINCIPAL DE RESPUESTA:
Para TODAS las preguntas, responde primero de manera CONCISA y DIRECTA (máximo 1-2 oraciones), luego pregunta si desea más detalles.

FORMATO DE RESPUESTA:
[RESPUESTA DIRECTA EN 1-2 ORACIONES]

¿Te gustaría que profundice en algún aspecto específico?

REGLAS ADICIONALES:
1. SIEMPRE responde en español
2. Si la pregunta NO está relacionada con el programa de crédito, responde: "Esta pregunta no está relacionada con tu programa de crédito."
3. NUNCA uses headers, emojis o formato markdown en la respuesta inicial
4. Mantén la respuesta inicial simple y conversacional

SOLO si el usuario pide "más detalles", "análisis completo" o algo similar, entonces proporciona el análisis extenso con headers y métricas.

TEMAS VÁLIDOS: Programa de crédito, clientes, cartera, órdenes, pagos, cobranza, ventas, pipeline, riesgo, ROI, métricas financieras."""

def summarize_merchant_data(data: Dict) -> str:
    """Resumir datos del merchant para contexto de GPT"""
    try:
        buyers = data.get('buyers', [])
        orders = data.get('orders', [])
        payouts = data.get('payouts', [])
        payment_links = data.get('payment_links', [])

        # Calcular métricas clave
        active_clients = [b for b in buyers if b.get('credit', {}).get('credit_limit', 0) > 0]
        total_credit_issued = sum(b.get('credit', {}).get('credit_limit', 0) for b in active_clients)
        total_credit_used = sum(b.get('credit', {}).get('credit_used', 0) for b in active_clients)
        utilization = (total_credit_used / total_credit_issued * 100) if total_credit_issued > 0 else 0

        # Análisis de pagos vencidos
        overdue_orders = [o for o in orders if o.get('payment_status') == 'due']
        total_overdue = sum(o.get('amount', 0) for o in overdue_orders)

        # Ingresos y pipeline
        total_revenue = sum(p.get('amount', 0) for p in payouts)
        total_pipeline = sum(l.get('cart_total', 0) for l in payment_links)

        summary = f"""
RESUMEN DEL PROGRAMA DE CRÉDITO:

MÉTRICAS GENERALES:
- Clientes activos: {len(active_clients)}
- Crédito total otorgado: ${total_credit_issued:,.2f}
- Crédito utilizado: ${total_credit_used:,.2f} ({utilization:.1f}% utilización)
- Crédito disponible: ${total_credit_issued - total_credit_used:,.2f}

PERFORMANCE FINANCIERA:
- Ingresos totales: ${total_revenue:,.2f}
- Total órdenes: {len(orders)}

ANÁLISIS DE RIESGO:
- Órdenes vencidas: {len(overdue_orders)}
- Monto en riesgo: ${total_overdue:,.2f}

PIPELINE DE VENTAS:
- Links pendientes: {len(payment_links)}
- Valor del pipeline: ${total_pipeline:,.2f}
"""

        return summary

    except Exception as e:
        return f"Error resumiendo datos: {str(e)}"

# Crear el chain de LangChain
def create_kontempo_chain():
    # Modelo
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3,
        api_key=OPENAI_API_KEY
    )
    
    # Template del prompt simplificado
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "ROL DEL USUARIO: {user_role}\n\nDATOS DEL MERCHANT:\n{merchant_summary}\n\nPREGUNTA DEL USUARIO: {query}")
    ])
    
    # Chain
    chain = prompt | llm | StrOutputParser()
    
    return chain

# Función para procesar input
def process_input(input_data: Dict) -> Dict:
    """Procesa el input y prepara el contexto"""
    query = input_data.get("query", "")
    merchant_data = input_data.get("context", {})
    user_context = input_data.get("user", {})
    conversation_history = input_data.get("conversation_history", [])
    
    # Convertir historial a mensajes de LangChain
    chat_history = []
    for msg in conversation_history[:-1]:  # Excluir el mensaje actual
        if msg["role"] == "user":
            chat_history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            chat_history.append(AIMessage(content=msg["content"]))
    
    # Preparar contexto
    merchant_summary = summarize_merchant_data(merchant_data)
    user_role = user_context.get("role", "admin")
    
    return {
        "query": query,
        "chat_history": chat_history,
        "merchant_summary": merchant_summary,
        "user_role": user_role
    }

# Crear la app FastAPI
app = FastAPI(
    title="Kontempo AI with LangChain",
    version="1.0",
    description="AI Assistant for Kontempo merchants using LangChain",
)

# Crear el chain
kontempo_chain = create_kontempo_chain()

# Agregar rutas con LangServe
add_routes(
    app,
    kontempo_chain,
    path="/kontempo",
    input_type=Dict,
    playground_type="chat",
)

# Endpoint personalizado compatible con tu frontend actual
@app.post("/chat")
async def chat_endpoint(request_data: Dict):
    try:
        # Procesar input
        processed_input = process_input(request_data)
        
        # Ejecutar chain
        response = await kontempo_chain.ainvoke(processed_input)
        
        return {
            "response": response,
            "timestamp": 1234567890,
            "model": "Kontempo AI with LangChain",
            "status": "success"
        }
    except Exception as e:
        return {
            "error": str(e),
            "status": "error"
        }

@app.post("/test")
async def test_endpoint(request_data: Dict):
    # Datos mock para testing
    mock_data = {
        "query": request_data.get("query", "¿Cómo va mi programa de crédito?"),
        "context": {
            "buyers": [
                {
                    "buyer_account": "buy_123",
                    "display_name": "Test Client",
                    "approval_status": "active",
                    "credit": {"credit_limit": 100000, "credit_used": 50000}
                }
            ],
            "orders": [
                {"buyer_account": "buy_123", "amount": 50000, "payment_status": "pristine"}
            ],
            "payouts": [
                {"amount": 45000, "payout_date": 1234567890}
            ],
            "payment_links": [
                {"cart_total": 25000, "expires": 1234567890}
            ]
        },
        "user": {"role": request_data.get("role", "admin")},
        "conversation_history": request_data.get("conversation_history", [])
    }
    
    return await chat_endpoint(mock_data)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)