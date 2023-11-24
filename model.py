# Import necessary libraries
import torch
# Llamaindex also works with langchain framework to implement embeddings 
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes, DecodingMethods
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models import Model

# Check for GPU availability and set the appropriate device for computation.
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

Watsonx_API = "Watsonx_API"
Project_id= "Project_id"

# Function to initialize the language model 
def init_llm():
    global llm, model
    
    params = {
        GenParams.MAX_NEW_TOKENS: 1024, # The maximum number of tokens that the model can generate in a single run.
        GenParams.MIN_NEW_TOKENS: 1,   # The minimum number of tokens that the model should generate in a single run.
        GenParams.DECODING_METHOD: DecodingMethods.SAMPLE, # The method used by the model for decoding/generating new tokens. In this case, it uses the sampling method.
        GenParams.TEMPERATURE: 0.6,   # A parameter that controls the randomness of the token generation. A lower value makes the generation more deterministic, while a higher value introduces more randomness.
        GenParams.TOP_K: 50,          # The top K parameter restricts the token generation to the K most likely tokens at each step, which can help to focus the generation and avoid irrelevant tokens.
        GenParams.TOP_P: 1            # The top P parameter, also known as nucleus sampling, restricts the token generation to a subset of tokens that have a cumulative probability of at most P, helping to balance between diversity and quality of the generated text.
    }
    
    credentials = {
        'url': "https://us-south.ml.cloud.ibm.com",
        'apikey' : Watsonx_API
    }

    model_id = ModelTypes.LLAMA_2_70B_CHAT
    
    model = Model(
        model_id= model_id,
        credentials=credentials,
        params=params,
        project_id=Project_id)

    llm = WatsonxLLM(model=model)

init_llm()
