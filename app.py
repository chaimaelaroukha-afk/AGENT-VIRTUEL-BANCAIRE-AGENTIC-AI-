import os
import time
from typing import Annotated, Sequence, TypedDict
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_groq import ChatGroq

# ==========================================
# 0. CONFIGURATION DE LA CLÉ API GROQ
# ==========================================
os.environ["GROQ_API_KEY"] = "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"

# ==========================================
# 1. PRÉTRAITEMENT ET VECTORISATION 
# ==========================================
print("⏳ Chargement et découpage de la base documentaire...")
loader = TextLoader("banque_data.txt", encoding="utf-8")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = Chroma.from_documents(docs, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 2})
print("✅ Base de données vectorielle prête (ChromaDB).")

# ==========================================
# 2. DÉVELOPPEMENT DES OUTILS 
# ==========================================
@tool
def outil_recherche_bancaire(query: str) -> str:
    """Recherche des informations officielles dans la base documentaire sur les tarifs, plafonds et règles de la banque."""
    print(f"   🔍 [Outil RAG] Recherche pour : '{query}'")
    documents_retrouves = retriever.invoke(query)
    if documents_retrouves:
        return "\n---\n".join([d.page_content for d in documents_retrouves])
    return "Aucune information trouvée dans les documents officiels."

@tool
def outil_calculateur_commission(montant: float) -> str:
    """Calcule automatiquement la commission bancaire standard de 2% sur une transaction financière."""
    print(f"   🧮 [Outil Calcul] Calcul de commission pour : {montant} DH")
    return f"La commission de 2% pour un montant de {montant} DH s'élève exactement à {montant * 0.02} DH."

liste_outils = [outil_recherche_bancaire, outil_calculateur_commission]
noeud_outils = ToolNode(liste_outils)

# ==========================================
# 3. ARCHITECTURE DU GRAPHE LANGGRAPH 
# ==========================================
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0).bind_tools(liste_outils)

INSTRUCTIONS_SYSTEME = SystemMessage(
    content=(
        "Tu es un conseiller bancaire virtuel de la Banque Centrale Populaire (BCP). "
        "Utilise obligatoirement les outils à ta disposition pour répondre aux requêtes. "
        "IMPORTANT : Dès qu'un outil (recherche ou calculateur) te renvoie un résultat, "
        "ne ré-invoque aucun outil. Prends cette information et formule immédiatement "
        "ta réponse finale en français de manière concise et professionnelle."
    )
)

def appeler_agent(state: AgentState):
    print("🤖 [Agent] Réflexion en cours...")
    messages_complets = [INSTRUCTIONS_SYSTEME] + list(state["messages"])
    reponse = llm.invoke(messages_complets)
    return {"messages": [reponse]}

workflow = StateGraph(AgentState)
workflow.add_node("agent", appeler_agent)
workflow.add_node("tools", noeud_outils)

workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", tools_condition)
workflow.add_edge("tools", "agent")

app = workflow.compile()
print("✅ Architecture LangGraph compilée avec succès !")

try:
    graph_png = app.get_graph().draw_mermaid_png()
    with open("graphe_architecture.png", "wb") as f:
        f.write(graph_png)
    print("🖼️ Graphe d'architecture exporté sous 'graphe_architecture.png' !")
except Exception:
    print("⚠️ Visualisation non générée.")

# ==========================================
# 4. SIMULATION ET TEST AUTOMATIQUE 
# ==========================================
if __name__ == "__main__":
    print("\n🚀 --- Début de la Grande Simulation d'Évaluation (20 Questions) ---")
    
    questions_test = [
        # --- 10 QUESTIONS SIMPLES ---
       
        "Quel est le plafond de retrait maximum autorisé par jour avec une carte Visa Classique de la Banque Centrale Populaire ?",
        "Quels sont les frais de tenue de compte appliqués par trimestre ?",
        "Quelle est la commission fixe pour un virement international standard ?",
        "Quel est le plafond d'achat mensuel avec une carte Visa Premier ?",
        "Quels sont les horaires d'ouverture de l'agence principale le samedi ?",
        "Quelles sont les pièces justificatives requises pour ouvrir un compte courant ?",
        "Quel est le taux d'intérêt annuel appliqué sur le Livret d'épargne ?",
        "À combien s'élèvent les frais de réédition d'un code secret de carte bancaire perdu ?",
        "Est-il possible d'effectuer un dépôt de chèque directement sur un automate externe ?",
        "Quel est le délai standard de réception d'une nouvelle carte bancaire après commande ?",
        
        # --- 10 QUESTIONS COMPLEXES ---
        "Calcule la commission pour un transfert de 150000 DH.",
        "Si je réalise un transfert de 45000 DH, quel montant exact de commission sera prélevé ?",
        "Un client souhaite transférer 80000 DH. Calcule sa commission et indique son plafond restant sur une Visa Classique.",
        "Je dois envoyer 12500 DH à un proche. Quel sera le coût total de l'opération en incluant la commission de 2% ?",
        "Calcule la commission totale facturée si j'effectue deux transferts de 60000 DH le même jour.",
        "Si un virement international a une commission de 2%, combine cela avec les frais fixes pour un montant de 200000 DH.",
        "Fais le calcul de la commission pour un transfert de 350000 DH and explique la procédure de conformité associée.",
        "Calcule la commission de 2% sur un dépôt exceptionnel de 12000 DH pour un compte soumis aux frais de gestion.",
        "Si je dépasse mon plafond d'achat de 5000 DH sur une Visa Premier, combien de frais de dépassement vais-je payer ?",
        "Calcule la commission pour un transfert de 95000 DH et indique si cela nécessite un rapport à l'Autorité Financière."
    ]
    
    config = {"recursion_limit": 10}
    
    for i, q in enumerate(questions_test, 1):
        type_question = "SIMPLE" if i <= 10 else "COMPLEXE"
        print(f"\n🔹 [{type_question}] Question {i} : '{q}'")
        inputs = {"messages": [HumanMessage(content=q)]}
        
        temps_debut = time.time()
        
        try:
            resultat_final = app.invoke(inputs, config=config)
            temps_fin = time.time()
            temps_reponse = temps_fin - temps_debut
            
            reponse_finale = resultat_final["messages"][-1].content
            print(f"💬 Réponse de l'Agent : {reponse_finale}")
            print(f"⏱️ Temps de réponse : {temps_reponse:.2f} secondes")
            
        except Exception as e:
            temps_fin = time.time()
            temps_reponse = temps_fin - temps_debut
            print("🏁 [Graphe] Interruption ou limite de sécurité atteinte.")
            try:
                state_actuel = app.get_state(inputs)
                messages = state_actuel.values.get("messages", [])
                if messages:
                    print(f"💬 Dernière réponse générée : {messages[-1].content}")
                print(f"⏱️ Temps écoulé jusqu'à l'interruption : {temps_reponse:.2f} secondes")
            except Exception:
                print("❌ Impossible de récupérer la réponse.")
                
        print("-" * 70)
        
    print("\n🎯 --- Fin de la Simulation  ---")