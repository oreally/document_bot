import streamlit as st

st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
    st.markdown(
        """
    <style>
        [data-testid="collapsedControl"] {
            display: none
        }
    </style>
    """,
        unsafe_allow_html=True,
    )

from documents import document_manager
from bot import chatbot


def main():
    
    col0, col1, col2, col3, col4 = st.columns([3, 30, 3, 61, 3])
    
    with col1:
        st.header("Document Manager")
        document_manager()

    with col3:
        st.header("Mixtral 8x7b Satellite Chatbot 🚀🤖")
        chatbot()

if __name__ == "__main__":
    main()

