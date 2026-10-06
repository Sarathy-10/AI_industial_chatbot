import streamlit as st

from rag import IndustrialRAG

import time

from textwrap import dedent



# ============================================================

# PAGE CONFIG

# ============================================================



st.set_page_config(

    page_title="AI Industrial Chatbot",

    page_icon="🏭",

    layout="wide"

)





# ============================================================

# CUSTOM CSS

# ============================================================



st.markdown("""

<style>



/* =========================================================

   MAIN BACKGROUND

   ========================================================= */



.stApp {

    background:

        radial-gradient(circle at 15% 15%, rgba(255, 145, 0, 0.06), transparent 30%),

        radial-gradient(circle at 85% 80%, rgba(0, 150, 255, 0.04), transparent 30%),

        #0e1117;

}





/* =========================================================

   ANIMATED TITLE

   ========================================================= */



.industrial-title {

    font-size: 3.2rem;

    font-weight: 800;



    background: linear-gradient(

        90deg,

        #ffffff,

        #ffb347,

        #ffffff,

        #ff8c42,

        #ffffff

    );



    background-size: 300% auto;



    -webkit-background-clip: text;

    -webkit-text-fill-color: transparent;



    animation: titleGradient 6s linear infinite;



    margin-bottom: 5px;

}



@keyframes titleGradient {



    0% {

        background-position: 0% center;

    }



    50% {

        background-position: 100% center;

    }



    100% {

        background-position: 0% center;

    }



}





/* =========================================================

   SUBTITLE

   ========================================================= */



.subtitle {

    color: #9ca3af;

    font-size: 1.05rem;

    margin-bottom: 25px;

}





/* =========================================================

   BOT ICON

   ========================================================= */



.bot-icon {

    display: inline-flex;



    width: 58px;

    height: 58px;



    align-items: center;

    justify-content: center;



    background: linear-gradient(

        135deg,

        #ff8a00,

        #ff5e00

    );



    border-radius: 16px;



    font-size: 30px;



    box-shadow:

        0 0 0 rgba(255, 136, 0, 0.5);



    animation: botPulse 2.5s infinite;



    margin-bottom: 12px;

}



@keyframes botPulse {



    0% {

        box-shadow: 0 0 0 0 rgba(255, 136, 0, 0.5);

        transform: scale(1);

    }



    50% {

        box-shadow: 0 0 25px 5px rgba(255, 136, 0, 0.15);

        transform: scale(1.04);

    }



    100% {

        box-shadow: 0 0 0 0 rgba(255, 136, 0, 0);

        transform: scale(1);

    }



}





/* =========================================================

   WELCOME CARD

   ========================================================= */



.welcome-card {



    padding: 24px;



    border-radius: 18px;



    background: linear-gradient(

        145deg,

        rgba(255,255,255,0.035),

        rgba(255,255,255,0.015)

    );



    border: 1px solid rgba(255,255,255,0.07);



    margin-bottom: 25px;



    animation: fadeUp 0.7s ease-out;

}





/* =========================================================

   MESSAGE ANIMATION

   ========================================================= */



[data-testid="stChatMessage"] {



    animation: messageAppear 0.45s ease-out;



}



@keyframes messageAppear {



    from {

        opacity: 0;

        transform: translateY(15px);

    }



    to {

        opacity: 1;

        transform: translateY(0);

    }



}





/* =========================================================

   CHAT MESSAGE HOVER

   ========================================================= */



[data-testid="stChatMessage"] {



    transition:

        transform 0.25s ease,

        background 0.25s ease;



}



[data-testid="stChatMessage"]:hover {



    transform: translateX(3px);



}





/* =========================================================

   THINKING ANIMATION

   ========================================================= */



.thinking-container {



    display: flex;



    align-items: center;



    gap: 8px;



    padding: 14px 18px;



    margin: 10px 0;



    border-radius: 14px;



    background: rgba(255,255,255,0.035);



    border: 1px solid rgba(255,255,255,0.06);



    color: #b8bec9;



    width: fit-content;



}





.thinking-dot {



    width: 8px;

    height: 8px;



    border-radius: 50%;



    background: #ff8a00;



    animation: thinking 1.4s infinite ease-in-out;



}



.thinking-dot:nth-child(2) {



    animation-delay: 0.2s;



}



.thinking-dot:nth-child(3) {



    animation-delay: 0.4s;



}





@keyframes thinking {



    0%, 60%, 100% {

        transform: translateY(0);

        opacity: 0.35;

    }



    30% {

        transform: translateY(-7px);

        opacity: 1;

    }



}





/* =========================================================

   SOURCE CARD

   ========================================================= */



.source-card {



    padding: 13px 16px;



    margin: 7px 0;



    border-radius: 12px;



    background: rgba(255,255,255,0.025);



    border: 1px solid rgba(255,255,255,0.06);



    transition:

        transform 0.2s ease,

        border-color 0.2s ease,

        background 0.2s ease;



}





.source-card:hover {



    transform: translateX(5px);



    background: rgba(255,255,255,0.045);



    border-color: rgba(255,145,0,0.35);



}





/* =========================================================

   SIDEBAR

   ========================================================= */



[data-testid="stSidebar"] {



    background:

        linear-gradient(

            180deg,

            #11151c,

            #0c0f14

        );



}





.sidebar-title {



    font-size: 1.25rem;



    font-weight: 700;



    color: white;



    margin-bottom: 10px;



}





/* =========================================================

   SIDEBAR DOCUMENT ITEMS

   ========================================================= */



.doc-item {



    padding: 10px 12px;



    margin: 6px 0;



    border-radius: 10px;



    background: rgba(255,255,255,0.025);



    border: 1px solid rgba(255,255,255,0.04);



    transition: all 0.2s ease;



}





.doc-item:hover {



    transform: translateX(4px);



    background: rgba(255,145,0,0.06);



    border-color: rgba(255,145,0,0.2);



}





/* =========================================================

   BUTTON

   ========================================================= */



.stButton > button {



    border-radius: 10px;



    transition:

        transform 0.2s ease,

        box-shadow 0.2s ease;



}





.stButton > button:hover {



    transform: translateY(-2px);



    box-shadow:

        0 5px 18px rgba(0,0,0,0.25);



}





/* =========================================================

   INPUT

   ========================================================= */



[data-testid="stChatInput"] {



    transition:

        transform 0.2s ease,

        box-shadow 0.2s ease;



}





[data-testid="stChatInput"]:focus-within {



    transform: translateY(-2px);



    box-shadow:

        0 0 25px rgba(255,140,0,0.12);



}





/* =========================================================

   FADE UP

   ========================================================= */



@keyframes fadeUp {



    from {



        opacity: 0;



        transform: translateY(18px);



    }



    to {



        opacity: 1;



        transform: translateY(0);



    }



}





/* =========================================================

   REDUCE ANIMATION FOR ACCESSIBILITY

   ========================================================= */



@media (prefers-reduced-motion: reduce) {



    * {



        animation-duration: 0.01ms !important;



        animation-iteration-count: 1 !important;



        transition-duration: 0.01ms !important;



    }



}



</style>

""", unsafe_allow_html=True)





# ============================================================

# LOAD RAG

# ============================================================



@st.cache_resource

def load_rag():



    return IndustrialRAG(load_llm=True)





rag = load_rag()





# ============================================================

# SESSION STATE

# ============================================================



if "messages" not in st.session_state:



    st.session_state.messages = []





# ============================================================

# SIDEBAR

# ============================================================



with st.sidebar:



    st.markdown(

        '<div class="sidebar-title">📚 Knowledge Base</div>',

        unsafe_allow_html=True

    )



    st.html("""
    <div class="doc-item">⚙️ IM-45 Induction Motor Maintenance</div>
    <div class="doc-item">🔧 PLC-200 Operations & Troubleshooting</div>
    <div class="doc-item">🏭 CNC-300 Operations & Safety</div>
    <div class="doc-item">⚡ Industrial Electrical Safety Guide</div>
    """)

    st.divider()



    st.markdown(

        '<div class="sidebar-title">🧠 Technologies</div>',

        unsafe_allow_html=True

    )



    st.write("🤗 Hugging Face Transformers")

    st.write("✨ Qwen 2.5 1.5B Instruct")

    st.write("🔎 Sentence Transformers")

    st.write("⚡ FAISS Vector Search")

    st.write("🎈 Streamlit")



    st.divider()



    if st.button(

        "🗑️ Clear Chat",

        use_container_width=True

    ):



        st.session_state.messages = []



        st.rerun()





# ============================================================

# HEADER

# ============================================================



st.html("""
<div class="industrial-header">
    <svg class="factory-icon" viewBox="0 0 96 96" aria-label="Animated industrial factory" role="img">
        <circle class="factory-smoke" cx="30" cy="15" r="6" fill="#ffb347"/>
        <circle class="factory-smoke s2" cx="36" cy="13" r="4.5" fill="#ffd08a"/>
        <circle class="factory-smoke s3" cx="43" cy="16" r="5" fill="#ffb347"/>

        <g class="factory-body">
            <path d="M8 78V45h18V32h13v13l15-10v10l14-9v42H8Z" fill="#ffb347"/>
            <rect x="8" y="78" width="72" height="5" rx="2.5" fill="#ffcf7a"/>

            <rect x="25" y="20" width="10" height="26" rx="2" fill="#ffb347"/>
            <rect x="43" y="28" width="8" height="18" rx="2" fill="#ffb347"/>

            <g fill="#14171d">
                <rect class="factory-window" x="15" y="53" width="9" height="9" rx="1.5"/>
                <rect class="factory-window" x="29" y="53" width="9" height="9" rx="1.5"/>
                <rect class="factory-window" x="43" y="53" width="9" height="9" rx="1.5"/>
                <rect class="factory-window" x="57" y="53" width="9" height="9" rx="1.5"/>
            </g>

            <g class="factory-gear" transform="translate(70 69)">
                <circle r="8" fill="#15181f"/>
                <circle r="3" fill="#ffb347"/>
                <path d="M0-12v5M0 7v5M-12 0h5M7 0h5M-8.5-8.5l3.5 3.5M5 5l3.5 3.5M8.5-8.5L5-5M-5 5l-3.5 3.5" stroke="#15181f" stroke-width="3" stroke-linecap="round"/>
            </g>
        </g>
    </svg>

    <div class="industrial-title">AI Industrial Chatbot</div>
</div>
""")



st.markdown(

    '<div class="subtitle">'

    'Document-grounded Industrial Assistant using Retrieval-Augmented Generation (RAG)'

    '</div>',

    unsafe_allow_html=True

)





# ============================================================
# WELCOME MESSAGE
# ============================================================

if len(st.session_state.messages) == 0:

    st.html("""
    <div class="welcome-card">
        <div class="bot-icon">🤖</div>
        <h3>👋 Hello! I'm your Industrial AI Assistant.</h3>
        <p>I can answer questions using the provided industrial documents covering:</p>
        <ul>
            <li>⚙️ Induction motor maintenance</li>
            <li>🔧 PLC troubleshooting</li>
            <li>🏭 CNC operations and safety</li>
            <li>⚡ Industrial electrical safety</li>
        </ul>
        <p><b>Ask me an industrial question to get started.</b></p>
    </div>
    """)




# DISPLAY PREVIOUS MESSAGES

# ============================================================



for message in st.session_state.messages:



    role = message["role"]



    with st.chat_message(role):



        st.markdown(message["content"])



        if (

            role == "assistant"

            and "sources" in message

            and message["sources"]

        ):



            with st.expander("📚 Retrieved Sources"):



                for source in message["sources"]:



                    st.html(f"""
                    <div class="source-card">
                        📄 <b>{source['source']}</b><br>
                        📖 Page: {source['page']}<br>
                        🔎 Similarity: {source['score']:.3f}
                    </div>
                    """)





# ============================================================

# CHAT INPUT

# ============================================================



question = st.chat_input(

    "Ask an industrial question..."

)





# ============================================================

# PROCESS QUESTION

# ============================================================



if question:



    # --------------------------------------------------------

    # USER MESSAGE

    # --------------------------------------------------------



    st.session_state.messages.append(

        {

            "role": "user",

            "content": question

        }

    )



    with st.chat_message("user"):



        st.markdown(question)





    # --------------------------------------------------------

    # AI MESSAGE

    # --------------------------------------------------------



    with st.chat_message("assistant"):



        thinking_placeholder = st.empty()

        thinking_placeholder.html("""
        <div class="thinking-container">
            <span>🤖 AI is thinking</span>
            <span class="thinking-dot"></span>
            <span class="thinking-dot"></span>
            <span class="thinking-dot"></span>
        </div>
        """)






        # ----------------------------------------------------

        # RAG QUERY

        # ----------------------------------------------------



        result = rag.ask(question)



        answer = result["answer"]



        sources = result["sources"]





        # Remove thinking animation



        thinking_placeholder.empty()





        # ----------------------------------------------------

        # ANSWER

        # ----------------------------------------------------



        st.markdown(answer)





        # ----------------------------------------------------

        # SOURCES

        # ----------------------------------------------------



        if sources:



            with st.expander("📚 Retrieved Sources"):



                for source in sources:



                    st.html(f"""
                    <div class="source-card">
                        📄 <b>{source['source']}</b><br>
                        📖 Page: {source['page']}<br>
                        🔎 Similarity: {source['score']:.3f}
                    </div>
                    """)





    # --------------------------------------------------------

    # SAVE ASSISTANT MESSAGE

    # --------------------------------------------------------



    st.session_state.messages.append(

        {

            "role": "assistant",

            "content": answer,

            "sources": sources

        }

    )