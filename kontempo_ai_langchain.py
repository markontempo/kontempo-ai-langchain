# kontempo_ai_langchain.py
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langserve import add_routes
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import List, Dict
import json

# Configuración
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Sistema de prompt
SYSTEM_PROMPT = """Eres un asistente especializado para usuarios del dashboard de Kontempo.

CONTEXTO:
- Kontempo es un SaaS para manejar programas de crédito
- Ayudas a merchants a analizar sus datos de clientes, órdenes y pagos

ESTRUCTURA DE DATOS:
- buyers[] = clientes con diferentes approval_status
- orders[] = transacciones realizadas
- payouts[] = pagos recibidos por el merchant
- payment_links[] = ventas pendientes

ANÁLISIS DE CLIENTES - STATUS DISPONIBLES:
Los buyers tienen estos approval_status posibles:
- "active": Cliente aprobado y activo para crédito
- "pending": Cliente en proceso de aprobación
- "rejected": Cliente rechazado para crédito
- "suspended": Cliente suspendido temporalmente

CAPACIDADES QUE TIENES:
✅ Listar clientes por status de aprobación
✅ Mostrar métricas de cartera y crédito
✅ Analizar performance de pagos
✅ Reportes de órdenes y cobranza
✅ Análisis de pipeline de ventas
✅ Segmentación de clientes activos/pendientes/rechazados

REGLAS DE RESPUESTA:
1. SIEMPRE responde en español
2. Para consultas de datos (listas, métricas), proporciona la información directamente
3. Sé conciso inicialmente, luego pregunta si quieren más detalles
4. SOLO rechaza preguntas no relacionadas con: clientes, crédito, ventas, pagos, cobranza, cartera

EJEMPLOS DE CONSULTAS VÁLIDAS:
- "Enlista los clientes por status"
- "¿Cuántos clientes activos tengo?"
- "Muéstrame los clientes pendientes de aprobación"
- "¿Cuál es mi cartera vencida?"
- "Lista de órdenes pagadas tarde"

FORMATO DE RESPUESTA PARA LISTAS:
Cuando te pidan listar clientes por status, organiza así:
**ACTIVOS (X):**
- Nombre Cliente 1
- Nombre Cliente 2

**PENDIENTES (X):**
- Nombre Cliente 3

**RECHAZADOS (X):**
- Nombre Cliente 4
"""

def summarize_merchant_data(data: Dict) -> str:
    try:
        buyers = data.get('buyers', [])
        
        # Agrupar clientes por status
        status_groups = {}
        for buyer in buyers:
            status = buyer.get('approval_status', 'unknown')
            if status not in status_groups:
                status_groups[status] = []
            status_groups[status].append({
                'name': buyer.get('display_name', 'Sin nombre'),
                'email': buyer.get('email', ''),
                'credit_limit': buyer.get('credit', {}).get('credit_limit', 0)
            })

        summary = f"""
RESUMEN DEL PROGRAMA DE CRÉDITO:

CLIENTES POR STATUS DE APROBACIÓN:
"""
        
        for status, clients in status_groups.items():
            status_name = {
                'active': 'ACTIVOS',
                'pending': 'PENDIENTES', 
                'rejected': 'RECHAZADOS',
                'suspended': 'SUSPENDIDOS'
            }.get(status, status.upper())
            
            summary += f"{status_name} ({len(clients)}):\n"
            for client in clients:
                summary += f"  • {client['name']}\n"
            summary += "\n"
        
        # Resto de métricas...
        orders = data.get('orders', [])
        payouts = data.get('payouts', [])
        
        total_revenue = sum(p.get('amount', 0) for p in payouts)
        summary += f"""
MÉTRICAS GENERALES:
- Total clientes: {len(buyers)}
- Ingresos totales: ${total_revenue:,.2f}
- Órdenes procesadas: {len(orders)}
"""
        
        return summary

    except Exception as e:
        return f"Error procesando datos: {str(e)}"

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

app = FastAPI(title="Kontempo AI", description="AI Assistant for Kontempo merchants")

# Agregar CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especifica dominios específicos
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
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
                "buyer_account": "buy_1234567890.1111",
                "display_name": "TECH INNOVATIONS SA DE CV",
                "email": "compras@techinnovations.mx",
                "approval_status": "active",
                "credit": {
                    "credit_limit": 120000,
                    "credit_used": 95000
                }
                },
                {
                "buyer_account": "buy_2345678901.2222",
                "display_name": "Materiales Industriales del Norte",
                "email": "pedidos@materiales-norte.com",
                "approval_status": "active",
                "credit": {
                    "credit_limit": 80000,
                    "credit_used": 55000
                }
                },
                {
                "buyer_account": "buy_3456789012.3333",
                "display_name": "CONSTRUCTORA MODERNA LTDA",
                "email": "admin@construmoderna.mx",
                "approval_status": "pending",
                "credit": {
                    "credit_limit": 0,
                    "credit_used": 0
                }
                },
                {
                "buyer_account": "buy_4567890123.4444",
                "display_name": "Distribuidora ABC",
                "email": "ventas@distribuidora-abc.com",
                "approval_status": "rejected",
                "credit": {
                    "credit_limit": 0,
                    "credit_used": 0
                }
                }
            ],
            "orders": [
                {
                "buyer_account": "buy_1234567890.1111",
                "amount": 45000,
                "payment_status": "completed_on_time",
                "created": 1734659051,
                "external_order_id": "ORD-001"
                },
                {
                "buyer_account": "buy_2345678901.2222", 
                "amount": 35000,
                "payment_status": "completed_late",
                "created": 1736285243,
                "external_order_id": "ORD-002"
                },
                {
                "buyer_account": "buy_1234567890.1111",
                "amount": 50000,
                "payment_status": "due",
                "created": 1737384627,
                "external_order_id": "ORD-003"
                }
            ],
            "payouts": [
                {
                "amount": 43650,
                "payout_date": 1738866140,
                "currency": "MXN",
                "status": "completed"
                },
                {
                "amount": 34300,
                "payout_date": 1739212444,
                "currency": "MXN", 
                "status": "completed"
                },
                {
                "amount": 49000,
                "payout_date": 1739816995,
                "currency": "MXN",
                "status": "pending"
                }
            ],
            "payment_links": [
                {
                "buyer_account": "buy_1234567890.1111",
                "cart_total": 50000,
                "expires": 1740700800,
                "external_order_id": "ORD-003",
                "description": "Materiales de construcción",
                "status": 0,
                "created": 1737395614
                },
                {
                "buyer_account": "buy_2345678901.2222",
                "cart_total": 25000,
                "expires": 1741305600,
                "external_order_id": "ORD-004", 
                "description": "Equipos industriales",
                "status": 0,
                "created": 1737482000
                }
            ]
            },
        "user": {"role": request_data.get("role", "admin")},
        "conversation_history": request_data.get("conversation_history", [])
    }
    
    return await chat_endpoint(mock_data)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
