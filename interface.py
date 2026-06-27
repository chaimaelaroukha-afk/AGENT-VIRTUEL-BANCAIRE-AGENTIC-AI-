import streamlit as st
from app import app 
from langchain_core.messages import HumanMessage
import os
os.environ["GROQ_API_KEY"] = "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"
st.title("🤖 Conseiller Virtuel - Banque Centrale Populaire")
st.subheader("Posez vos questions sur les tarifs et les commissions BCP")

# Initialisation de l'historique des messages dans la session du navigateur
if "messages" not in st.session_state:
    st.session_state.messages = []

# Affichage des messages passés
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Zone de saisie utilisateur
if prompt := st.chat_input("Comment puis-je vous aider aujourd'hui ?"):
    # Afficher le message de l'utilisateur
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Exécution de l'agent LangGraph
    with st.chat_message("assistant"):
        with st.spinner("L'agent réfléchit et interroge les outils..."):
            inputs = {"messages": [HumanMessage(content=prompt)]}
            config = {"recursion_limit": 10}

            # On appelle ton graphe
            resultat = app.invoke(inputs, config=config)
            reponse_finale = resultat["messages"][-1].content

            st.write(reponse_finale)
            st.session_state.messages.append({"role": "assistant", "content": reponse_finale})