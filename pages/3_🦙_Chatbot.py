import streamlit as st
import replicate


#Llama page configuration
st.set_page_config(page_title="Llama2Llama", page_icon=":llama:", layout="wide")

#Use local css
def local_css(file_name):
	with open(file_name) as f:
		st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style/style.css")

with st.container():
    st.title('🦙💬 Llama2 Chatbot')
    st.write("To talk to me you\'ll need a Replicate API key!")
    replicate_api = st.text_input('Enter Replicate API token:', type='password')
    if not (replicate_api.startswith('r8_') and len(replicate_api)==40):
        st.warning('I need a key!', icon='🗝')

with st.container():
    st.subheader('Parameters')
    temperature = st.slider('temperature', min_value=.01, max_value=5.0, value=.1, step=.01)
    top_p = st.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
    max_length = st.slider('max_length', min_value=32, max_value=128, value=120, step=8)


#Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

#Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

#Function for generating Llama2 response
def generate_response(prompt_input):
    string_dialogue = "You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'."
    for dict_message in st.session_state.messages:
        if dict_message["role"] == "user":
            string_dialogue += "User: " + dict_message["content"] + "\n\n"
        else:
            string_dialogue += "Assistant: " + dict_message["content"] + "\n\n"
    output = replicate.run("meta/llama-2-70b-chat:2c1608e18606fad2812020dc541930f2d0495ce32eee50074220b87300bc16e1", input={"prompt": f"{string_dialogue} {prompt_input} Assistant: ", 
                                       "temperature":temperature, "top_p":top_p, "max_length":max_length, "repetition_penalty":1})
    return output

#User-provided prompt
if prompt := st.chat_input(disabled=not replicate_api):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_response(prompt)
            placeholder = st.empty()
            full_response = ''
            for item in response:
                full_response += item
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)

    
    
    
    