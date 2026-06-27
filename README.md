# AGENT-VIRTUEL-BANCAIRE-AGENTIC-AI-
Ce projet met en lumière la conception, l'évaluation et la mise en œuvre d'un Agent Virtuel Bancaire intelligent basé sur une structure Agentic RAG. Cet agent repousse les frontières des chatbots conventionnels en intégrant la recherche documentaire sémantique avec le raisonnement autonome assisté par un graphe d'états
-Fonctionnalités Clés :
-Raisonnement Agentique (LangGraph) : Implémentation de bas niveau d'un graphe orienté et cyclique pour orchestrer les décisions de l'agent sans utiliser de méthode pré-construite.
-Recherche Sémantique (RAG) : Indexation de la base de connaissances métiers (banque_data.txt : tarifs, plafonds, horaires) dans un magasin de vecteurs local ChromaDB avec les embeddings all-MiniLM-L6-v2.  
-Calculs Déterministes : Intégration d'outils sur mesure (tool_calls), notamment un calculateur arithmétique pour évaluer les commissions bancaires exactes de 2% sans risque d'hallucination du LLM. 
-Interface Web ChatGPT-like (Streamlit) : Une application web interactive et fluide intégrant un historique de session et un indicateur visuel de réflexion de l'agent.  
🛠️- Technologies UtiliséesFrameworks : LangGraph, LangChain 
Modèle de Langage (LLM) : llama-3.1-8b-instant (via l'infrastructure Groq) 
Base de Données Vectorielle : ChromaDB 
Interface Utilisateur : Streamlit  
Langage : Python


1-INTERFACE VS CODE 
<img width="1842" height="850" alt="image" src="https://github.com/user-attachments/assets/bb758bd2-db8a-4d5c-9559-7fd9880d7d39" />

2-INTERFACE WEB STREAMLIT
<img width="1867" height="947" alt="image" src="https://github.com/user-attachments/assets/a4ea03b8-1d44-4193-bcb3-0ff65270b7b8" />



