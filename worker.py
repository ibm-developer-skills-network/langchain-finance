import torch
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes, DecodingMethods
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models import Model
import yfinance as yf
import os
from langchain.document_loaders.web_base import WebBaseLoader
from langchain.utilities import GoogleSerperAPIWrapper
from langchain_experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner
from langchain.agents.tools import Tool
from langchain import LLMChain
from langchain.prompts import PromptTemplate
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

Watsonx_API = "Watsonx_API"
Project_id= "Project_id"

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

os.environ["SERPER_API_KEY"] = "SERPER_API_KEY"

def get_ticker_news(ticker, n_search_results=3):
    # Get the news from Yahoo Finance
    links = []
    try:
        company = yf.Ticker(ticker) # Get the yfinance.Ticker object
        links = [n["link"] for n in company.news if n["type"] == "STORY"] # Store links for later use
        print(f"Links stored successfully!")
    except: 
        print(f"No news found from Yahoo Finance.")
    
    print(yf.Ticker(ticker).info['longName'])
    company = yf.Ticker(ticker).info['longName'] 
    search = GoogleSerperAPIWrapper(type="news", tbs="qdr:d5", serper_api_key=os.environ["SERPER_API_KEY"])
    results = search.results(f"financial news about {company} or {ticker}")
    if not results['news']:
        logger.error(f"No search results for the previous 5 days. Let's try to broaden the search range to one week!")
        search = GoogleSerperAPIWrapper(type="news", tbs="qdr:w1", serper_api_key=os.environ["SERPER_API_KEY"])
        results = search.results(f"financial news about {company} or {ticker}")
        print(f"News found from Google News.")
        if not results['news']:
            logger.error(f"No news found from Google News.")
    else:
        for i, item in zip(range(n_search_results), results['news']):
            try:
                links.append(item['link'])
                print(f"{i+1} links stored successfully!")
            except: 
                print(f"Exceed the number of search results.")

    loader = WebBaseLoader(links)
    docs = loader.load()
    data = _format_results(docs, ticker)  
    print(f"Completed getting news for {ticker}")
    return data

def _format_results(docs, query):
    title_content_list = []

    for doc in docs:
        title = doc.metadata.get('title', 'No title available')
        page_content = doc.page_content.strip() if query in doc.page_content else "" # Remove leading/trailing spaces
        title_content = f"{title}:{page_content}\n"
        title_content_list.append(title_content)

    return "\n".join(title_content_list)

def get_financial_statements(ticker):
    company = yf.Ticker(ticker)
    
    balance_sheet = company.balance_sheet
    cash_flow = company.cash_flow
    income_statement = company.income_stmt
    
    csv_file_prefix = f"{ticker}_financial_"
    stock_data = yf.download(ticker, period='5y', interval='1d')
    data_csv_filename = csv_file_prefix + "stock_data.csv"
    stock_data.to_csv(data_csv_filename)
    
    balance_sheet_csv_filename = csv_file_prefix + "balance_sheet.csv"
    cash_flow_csv_filename = csv_file_prefix + "cash_flow.csv"
    income_statement_csv_filename = csv_file_prefix + "income_statement.csv"
    valuation_measures_csv_filename = csv_file_prefix + "valuation_measures.csv"
    
    balance_sheet.to_csv(balance_sheet_csv_filename)
    cash_flow.to_csv(cash_flow_csv_filename)
    income_statement.to_csv(income_statement_csv_filename)
    
    print('Stock price data and financial statements are saved to the CSV files')
    return data_csv_filename, balance_sheet_csv_filename, cash_flow_csv_filename, income_statement_csv_filename

tools = [  
    Tool(
        name="Get Recent News",
        func=get_ticker_news,
        description="Useful when you want to obtain information about current financial events."
    ), 
    Tool(
        name = "Get Financial Statements",
        func=get_financial_statements,
        description="Useful when you need to analyze the company's stock price and financial statements to find more insight. Input should be a company ticker such as TSLA for Tesla, NVDA for NVIDIA."
    ),
]

template = """
<s>[INST] <<SYS>>
As a stock investment advisor, your role is to provide investment recommendations based on the company's current financial performance and market trends.
You refrain from providing direct 'Buy' or 'Sell' recommendations to comply with legal regulations. 
Utilize the data in your database to create a detailed investment thesis to address the user's request.
Please back your assertions with substantial data and analysis.

Use the following format:
Insights from News: share the insights found from the recent news. Be sure to mention the specific time frame or period you are analyzing. 
<br> 
Balance Sheet Analysis: provide an analysis of the company's balance sheet. Be sure to mention the specific time frame or period you are analyzing. 
<br> 
Cash Flow Analysis: write an analysis of its cash flow to answer user query. Be sure to mention the specific time frame or period you are analyzing. 
<br> 
Income Statement Analysis: write an analysis of its income statement to answer user query. Be sure to mention the specific time frame or period you are analyzing. 
<br> 
Summary: the final answer to the original input question backed up by the insights and analysis above. Be sure to mention the specific time frame or period you are analyzing. 
<br> 
Note: Anything else relevant to the ticker. You should also encourage the user to conduct further research and consider various factors before making any investment decisions.

Begin!
Question: {input}
{agent_scratchpad}
<</SYS>>
[/INST]
"""

prompt = PromptTemplate(input_variables=["input", "agent_scratchpad"], template=template)
llm_chain = LLMChain(
    llm=llm,
    prompt=prompt
)

tool_names = [tool.name for tool in tools]
agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names)
agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=False, max_iterations = 2)

def process_prompt(ticker):
    try:
        ticker = ticker.upper()
        response = agent_executor.run(input=ticker)
    except ValueError as e:
        response = str(e)
        if not response.startswith("Could not parse LLM output: `"):
            raise e
        response = response.removeprefix("Could not parse LLM output: `").removesuffix("`")

    return {
        'input': ticker,
        'text': response
    }
