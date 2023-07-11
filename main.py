import os

import streamlit as st
from dotenv import load_dotenv
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from streamlit_chat import message

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

load_dotenv(".env_file")


def load_chain():
    """Logic for loading the chain you want to use should go here."""
    llm = LlamaCpp(
        model_path=os.environ.get(
            "LLM_MODEL_PATH", "./orca-mini-v2-ger-7b.ggmlv3.q4_0.bin"
        ),
        callback_manager=callback_manager,
        verbose=True,
        n_gpu_layers=1,
        # n_batch=512,
    )
    template = "### System:\n{system}\n\n### User:\n{instruction}\n\n### Response:\n"
    prompt = PromptTemplate(
        template=template, input_variables=["system", "instruction"]
    )
    prompt_w_system = prompt.partial(
        system="You are an AI assistant that follows instruction extremely well. Help as much as you can. If the user speaks German, please also answer in German."
    )
    chain = LLMChain(prompt=prompt_w_system, llm=llm)
    return chain


chain = load_chain()

# From here down is all the StreamLit UI.
st.set_page_config(page_title="Demo", page_icon=":robot:")
st.header("Demo")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []


def get_text():
    input_text = st.text_input("You: ", "Hallo, wie gehts?", key="input")
    return input_text


user_input = get_text()

if user_input:
    output = chain.run(instruction=user_input)

    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state["generated"]:
    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
