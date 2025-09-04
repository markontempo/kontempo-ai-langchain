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
                "buyer_account": "buy_6978320898.8846",
                "display_name": "DIBAR NUTRICIONAL SRL DE CV",
                "email": "direccion@laboratoriosdibar.com",
                "phone": "+18324999960",
                "approval_status": "active",
                "credit": {
                    "credit_limit": 0,
                    "credit_used": 0
                },
                "address": {
                    "street": "AV TRES MARIAS",
                    "external_number": "455",
                    "internal_number": "17",
                    "area": "Orquídeas",
                    "city": "Morelia",
                    "state": "Michoacán de Ocampo",
                    "postal_code": "58254",
                    "country": "Mexico"
                },
                "rfc_id": "DNU1306241HA",
                "foundation_date": 2013,
                "order_count": 167
                },
                {
                "buyer_account": "buy_6796692618.2490",
                "display_name": "Primeprotein",
                "email": "osnaya.adriana.mercado@gmail.com",
                "phone": "+525531407018",
                "approval_status": "rejected",
                "credit": {
                    "credit_limit": 0,
                    "credit_used": 0
                },
                "address": {
                    "street": "Jilguero",
                    "external_number": "65",
                    "area": "Bellavista",
                    "city": "Álvaro Obregón",
                    "state": "Ciudad de México",
                    "postal_code": "01140",
                    "country": "Mexico"
                },
                "rfc_id": "PNH140311GZ4",
                "foundation_date": 2014,
                "sub_status": "credit_analysis"
                },
                {
                "buyer_account": "buy_6540029417.3229",
                "display_name": "temp_1d1928311f484c85ad8e4986919dc98d",
                "email": "compras@healthylab.com.mx",
                "phone": "+528445426883",
                "approval_status": "pending",
                "credit": {
                    "credit_limit": 0,
                    "credit_used": 0
                },
                "rfc_id": "RFC-00000",
                "sub_status": "additional_information_required"
                },
                {
                "buyer_account": "buy_2693382568.6016",
                "display_name": "Laboratorios Solfran",
                "email": "compras2@solfran.com",
                "phone": "+523335703879",
                "approval_status": "pending",
                "credit": {
                    "credit_limit": 0,
                    "credit_used": 0
                },
                "address": {
                    "street": "CALLE ALTOS HORNOS",
                    "external_number": "2721",
                    "area": "El Álamo",
                    "city": "San Pedro Tlaquepaque",
                    "state": "Jalisco",
                    "postal_code": "45560",
                    "country": "Mexico"
                },
                "rfc_id": "LSO741128J68",
                "foundation_date": 1974,
                "requested_amount": 200000,
                "annual_revenue_current_year": 480000000,
                "sub_status": "additional_information_required"
                },
                {
                "buyer_account": "buy_8798244248.1389",
                "display_name": "INSTITUTO DE INVESTIGACION BIOTECNOLOGICA COSMETICA",
                "email": "administracion@iinbic.com",
                "phone": "+525513046514",
                "approval_status": "rejected",
                "credit": {
                    "credit_limit": 0,
                    "credit_used": 0
                },
                "address": {
                    "street": "GALEANA",
                    "external_number": "144",
                    "area": "Santa Fe",
                    "city": "Álvaro Obregón",
                    "state": "Ciudad de México",
                    "postal_code": "01210",
                    "country": "Mexico"
                },
                "rfc_id": "IIB160127RP1",
                "foundation_date": 2016,
                "requested_amount": 50000,
                "annual_revenue_current_year": 3000000,
                "sub_status": "additional_information_required"
                }
            ],
            "orders": [
                {
                "buyer_account": "buy_6978320898.8846",
                "amount": 320000,
                "payment_status": "completed_late",
                "created": 1734659051,
                "external_order_id": "COT #28689",
                "merchant_account": "mer_1733158748.482",
                "currency": "MXN",
                "status": "completed",
                "payout_amount": 313505.48
                },
                {
                "buyer_account": "buy_2186917711.4727",
                "amount": 8014,
                "payment_status": "completed_late",
                "created": 1736285243,
                "external_order_id": "28842",
                "merchant_account": "mer_1733158748.482",
                "currency": "MXN",
                "status": "completed",
                "payout_amount": 7851.32
                },
                {
                "buyer_account": "buy_2186917711.4727",
                "amount": 11100,
                "payment_status": "completed_late",
                "created": 1736522058,
                "external_order_id": "28920",
                "merchant_account": "mer_1733158748.482",
                "currency": "MXN",
                "status": "completed",
                "payout_amount": 10874.67
                },
                {
                "buyer_account": "buy_4139016637.5655",
                "amount": 2750,
                "payment_status": "completed_on_time",
                "created": 1736956134,
                "external_order_id": "OV 24292",
                "merchant_account": "mer_1733158748.482",
                "currency": "MXN",
                "status": "completed",
                "payout_amount": 2694.18
                },
                {
                "buyer_account": "buy_2186917711.4727",
                "amount": 10500,
                "payment_status": "completed_late",
                "created": 1737384627,
                "external_order_id": "24368",
                "merchant_account": "mer_1733158748.482",
                "currency": "MXN",
                "status": "completed",
                "payout_amount": 10286.85
                },
                {
                "buyer_account": "buy_6650742974.9102",
                "amount": 12900,
                "payment_status": "completed_on_time",
                "created": 1737412541,
                "external_order_id": "24376",
                "merchant_account": "mer_1733158748.482",
                "currency": "MXN",
                "status": "completed",
                "payout_amount": 12638.17
                },
                {
                "buyer_account": "buy_6650742974.9102",
                "amount": 23494.49,
                "payment_status": "completed_on_time",
                "created": 1737412890,
                "external_order_id": "24372",
                "merchant_account": "mer_1733158748.482",
                "currency": "MXN",
                "status": "completed",
                "payout_amount": 23017.63
                },
                {
                "buyer_account": "buy_6650742974.9102",
                "amount": 301888.9,
                "payment_status": "completed_on_time",
                "created": 1737413029,
                "external_order_id": "24373",
                "merchant_account": "mer_1733158748.482",
                "currency": "MXN",
                "status": "completed",
                "payout_amount": 295761.52
                },
                {
                "buyer_account": "buy_6650742974.9102",
                "amount": 38334.8,
                "payment_status": "completed_on_time",
                "created": 1737413101,
                "external_order_id": "24377",
                "merchant_account": "mer_1733158748.482",
                "currency": "MXN",
                "status": "completed",
                "payout_amount": 37556.73
                },
                {
                "buyer_account": "buy_4139016637.5655",
                "amount": 4926.74,
                "payment_status": "completed_on_time",
                "created": 1737424165,
                "external_order_id": "OV 24395",
                "merchant_account": "mer_1733158748.482",
                "currency": "MXN",
                "status": "completed",
                "payout_amount": 4826.73
                }
            ],
            "payouts": [
                {
                "amount": 7851.32,
                "payout_date": 1738866140,
                "merchant_account": "mer_1733158748.482",
                "currency": "MXN",
                "status": "completed",
                "payout_id": "payout_92a0783fed59e5db4c44aa5cf2379e11"
                },
                {
                "amount": 10874.67,
                "payout_date": 1739212444,
                "merchant_account": "mer_1733158748.482",
                "currency": "MXN",
                "status": "completed",
                "payout_id": "payout_8b2424d1b9e4e76a77012cbe2ed30af9"
                },
                {
                "amount": 2694.18,
                "payout_date": 1739557127,
                "merchant_account": "mer_1733158748.482",
                "currency": "MXN",
                "status": "completed",
                "payout_id": "payout_c170ebd6770a3593770419a5f27483d5"
                },
                {
                "amount": 313505.48,
                "payout_date": 1739816995,
                "merchant_account": "mer_1733158748.482",
                "currency": "MXN",
                "status": "completed",
                "payout_id": "payout_d2c801b24087227d782cd0853a61d8bb"
                },
                {
                "amount": 15113.58,
                "payout_date": 1739989107,
                "merchant_account": "mer_1733158748.482",
                "currency": "MXN",
                "status": "completed",
                "payout_id": "payout_767d990ed2e4f89052896c445e4df049"
                },
                {
                "amount": 11085.25,
                "payout_date": 1740075395,
                "merchant_account": "mer_1733158748.482",
                "currency": "MXN",
                "status": "completed",
                "payout_id": "payout_f07be59b73d901d3afafed6621491d9f"
                },
                {
                "amount": 382771.76,
                "payout_date": 1741025804,
                "merchant_account": "mer_1733158748.482",
                "currency": "MXN",
                "status": "completed",
                "payout_id": "payout_40767a31f81076f8b842d6f0e80c6d9c"
                }
            ],
            "payment_links": [
                {
                "buyer_account": "buy_6978320898.8846",
                "cart_total": 320000,
                "expires": 1735603200,
                "external_order_id": "COT #28689",
                "merchant_account": "mer_1733158748.482",
                "currency": "MXN",
                "description": "MATERIAS PRIMAS",
                "status": 0,
                "created": 1734650256
                },
                {
                "buyer_account": "buy_2186917711.4727",
                "cart_total": 8014,
                "expires": 1738281600,
                "external_order_id": "28842",
                "merchant_account": "mer_1733158748.482",
                "currency": "MXN",
                "description": "20 Kg de Glucosamina malla 40",
                "status": 0,
                "created": 1736285076
                },
                {
                "buyer_account": "buy_2186917711.4727",
                "cart_total": 11100,
                "expires": 1738281600,
                "external_order_id": "28920",
                "merchant_account": "mer_1733158748.482",
                "currency": "MXN",
                "description": "MATERIA PRIMA",
                "status": 0,
                "created": 1736521942
                },
                {
                "buyer_account": "buy_4139016637.5655",
                "cart_total": 2750,
                "expires": 1739577600,
                "external_order_id": "OV 24292",
                "merchant_account": "mer_1733158748.482",
                "currency": "MXN",
                "description": "YERBAMATE",
                "status": 0,
                "created": 1736955367
                },
                {
                "buyer_account": "buy_6650742974.9102",
                "cart_total": 301888.9,
                "expires": 1740700800,
                "external_order_id": "24373",
                "merchant_account": "mer_1733158748.482",
                "currency": "MXN",
                "description": "OC P0037",
                "status": 0,
                "created": 1737395614
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
