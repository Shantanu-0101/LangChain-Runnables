from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda, RunnableBranch

load_dotenv()

def word_counter(text):
    return len(text.split())

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-4-31B-it",
    task='text-generation'
)

model = ChatHuggingFace(llm=llm)

prompt1 = PromptTemplate(
    template='write a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Summarize the following text \n {text}',
    input_variables=['text']
)

parser = StrOutputParser()

report_gen_chain = RunnableSequence(prompt1 | model | parser)

branch_chain = RunnableBranch(
    (lambda x: len(x.split())>300, RunnableSequence(prompt2, model, parser)),
    RunnablePassthrough()
)

final_chain = RunnableSequence(report_gen_chain | branch_chain)

print(final_chain.invoke({'topic':'Russia vs Ukraine'}))