from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence

load_dotenv()

prompt = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-4-31B-it",
    task='text-generation',
    max_new_tokens=512
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

prompt2 = PromptTemplate(
    template='Explain the following joke \n {text}',
    input_variables=['text']
)


chain = RunnableSequence(prompt, model, parser, prompt2, model, parser)

print(chain.invoke({'topic':'AI'}))