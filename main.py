import os

import streamlit as st
from dotenv import load_dotenv
from langchain import LLMChain, PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
from langchain.llms import LlamaCpp
from langchain.schema import ChatMessage

load_dotenv()


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


if "system_message" not in st.session_state:
    st.session_state[
        "system_message"
    ] = "You are an AI assistant that follows instruction extremely well. Help as much as you can. If the user speaks German, please also answer in German."

st.text_input("System Message", key="system_message")

st.chat_message("assistant").write("Wie kann ich helfen?")

if "chain" not in st.session_state:
    stream_handler = StreamHandler(st.empty())
    llm = LlamaCpp(
        model_path=os.environ.get(
            "LLM_MODEL_PATH", "./orca-mini-v2-ger-7b.ggmlv3.q4_0.bin"
        ),
        callbacks=[stream_handler],
        verbose=False,
        n_gpu_layers=1,
        n_ctx=2048,
        use_mlock=True,
        temperature=0.7,
        top_p=1.0,
        top_k=50,
        max_tokens=1024,
        repeat_penalty=1.1,
    )  # type: ignore
    template = "### System:\n{system}\n\n### User:\n{instruction}\n\n### Response:\n"
    prompt = PromptTemplate(
        template=template, input_variables=["system", "instruction"]
    )
    prompt_w_system = prompt.partial(
        system="You are an AI assistant that follows instruction extremely well. Help as much as you can. If the user speaks German, please also answer in German."
    )
    llm_chain = LLMChain(prompt=prompt_w_system, llm=llm)
    st.session_state["chain"] = llm_chain


def update_model_parameters(temperature, top_p, top_k, repeat_penalty, max_tokens):
    st.session_state["chain"].llm.temperature = temperature
    st.session_state["chain"].llm.top_p = top_p
    st.session_state["chain"].llm.top_k = top_k
    st.session_state["chain"].llm.repeat_penalty = repeat_penalty
    st.session_state["chain"].llm.max_tokens = max_tokens


with st.sidebar:
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        key="temperature",
    )
    top_p = st.slider(
        "top_p", min_value=0.5, max_value=1.0, value=1.0, step=0.05, key="top_p"
    )
    top_k = st.slider(
        "top_k", min_value=10, max_value=80, value=50, step=10, key="top_k"
    )
    repeat_penalty = st.slider(
        "repeat_penalty",
        min_value=0.0,
        max_value=2.0,
        value=1.1,
        step=0.1,
        key="repeat_penalty",
    )
    max_tokens = st.slider(
        "max_tokens",
        min_value=128,
        max_value=1024,
        value=1024,
        step=128,
        key="max_tokens",
    )
    st.button(
        "Update Model Parameter",
        key="update",
        on_click=update_model_parameters(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            max_tokens=max_tokens,
        ),
    )

# input_placeholder=st.empty()

if prompt := st.chat_input():
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        stream_handler = StreamHandler(st.empty())
        st.session_state["chain"].llm.callbacks[0] = stream_handler
        response = st.session_state["chain"].run(prompt)
        # st.chat_message('Assistant').write(response)
