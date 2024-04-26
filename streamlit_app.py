import streamlit as st
import base64 
import os
from dotenv import load_dotenv
from openai import OpenAI
import tempfile

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

client = OpenAI()

sample_prompt = """You are a medical practitioner and an expert in analyzing medical related images
working for a very reputed hospital.  You will be provided with images and you need to identify 
the anomalies, any disease or health issues.  You need to generate the result in detailed manner. Write
all the findings, next steps, recommendations, etc.  You only need to respond if the image is related to a human
body and health issues.  You must have an answer but also write a disclaimer saying that "Consult
a doctor before making any decisions".

Remember: If certain aspect is not clear from the image, it is okay to state 'Unable to determin
based on the provided image.'

Now analyze the image and answer the above questions in teh ssame structured manned defined above.
"""

if 'uploaded_file'  not in st.session_state:
    st.session_state.uploaded_file = None
if 'result' not in st.session_state:
    st.session_state.result = None

def encode_image(image_path):
    with open(image_path,'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def call_gpt4_model_for_analysis(filename:str, sample_prompt=sample_prompt):
    base64_image = encode_image(filename)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text", "text": sample_prompt
                },
                {
                    "type": "image_url",
                    "image_url":{
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": "high"
                    }
                }
            ]
        }
    ]

    response = client.chat.completions.create(
        model = "gpt-4-turbo",
        messages = messages,
        max_tokens= 1100
    )

    print(response.choices[0].message.content)
    return response.choices[0].message.content

def chat_eli5(query):
    eli5_prompt = "You have to explain the below piece of information to a five year old. \n" + query
    messages = [
        {
            "role": "user",
            "content": eli5_prompt
        }
    ]
    response = client.chat.completions.create(
        model ="gpt-3.5-turbo",
        messages=messages,
        max_tokens=1100
    )

    return response.choices[0].message.content

st.title("Medical Image Analysis")

with st.expander("About the App"):
    st.write("Upload an image to get an analysis from GPT-4V")

uploaded_file = st.file_uploader('Upload your image', type=['jpg','png','jpeg'])  

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        st.session_state['filename']=tmp_file.name

    st.image(uploaded_file, caption="Uploaded Image")

if st.button('Analyze Image'):
    if 'filename' in st.session_state and os.path.exists(st.session_state['filename']):
        st.session_state['result']=call_gpt4_model_for_analysis(st.session_state['filename'])
        st.markdown(st.session_state['result'], unsafe_allow_html=True)
        os.unlink(st.session_state['filename'])

if 'result' in st.session_state and st.session_state['result']:
    st.info('Below is your result for ELI5 to understand in simpler terms')

    if st.radio("ELI5: Explain like I am 5", ('No','Yes')) == 'Yes':
        simplified_result = chat_eli5(st.session_state['result'])
        st.markdown(simplified_result, unsafe_allow_html=True)

