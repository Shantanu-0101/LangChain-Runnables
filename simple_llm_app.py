from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate

# Initialize the LLM
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-4-31B-it",
    task='text-generation',
    max_new_tokens=512
)

model = ChatHuggingFace(llm=llm)

#create a prompt template
prompt = PromptTemplate(
    input_variables=['topic'],
    template="Suggest a atchy blog tile about {topic}. "
)

#Define the topic
topic = input("Enter the topic")

# Format the prompt manually sing PromptTemplate
formatted_prompt = prompt.format(topic=topic)

# Call the LLM directly
blog_title = llm.predict(formatted_prompt)

#print the output
print("Generated Blog Title:", blog_title)