# Import the required modules
from langchain_google_genai import GoogleGenerativeAI  # Use to access the Google Generative AI
from langchain import PromptTemplate  # Use to create a prompt template
from langchain.chains import LLMChain  # Use to create a chain of LLMs
from langchain.chains import SequentialChain  # Use to create a chain (Sequential) of LLMs


import streamlit as st  # Use to create the web app


import os
import dotenv


dotenv.load_dotenv()


# the icon and the title of the web app
st.set_page_config(page_title="PROJECT-1 CELEBRITY STALKER", page_icon="🌟",layout="wide")




# streamlit framework
st.title('Celebrity Stalker AI 🌟')
input_text = st.text_input("Name a celebrity")


# Gemini LLMS
google_api_key = os.getenv("GOOGLE_GEMINI_AI")  # Google API Key
llm = GoogleGenerativeAI(temperature=0.8, google_api_key=google_api_key, model="gemini-pro")  # Initialize the Gemini LLM


# First Prompt Templates
first_input_prompt = PromptTemplate(
   input_variables=['name'],  # Input variables for creating the prompt template (Here name is the input variable)
   template="tell me about  {name}in 3 lines"
)


# Chain of LLMs
chain = LLMChain(  # First Chain Which uses the first prompt template to find about the person
   llm=llm,  # Pass the LLM (For our case it is Gemini LLM)
   prompt=first_input_prompt,  # Pass the prompt template
   verbose=True,  # Use verbose to print the output
   output_key='person'  # Output key to store the output of the LLM (person is the output key for the first chain so person will contain information about the person)
)


# Second Prompt Templates
second_input_prompt = PromptTemplate(  # Second Prompt Template
   input_variables=['person'],  # Input variables for creating the prompt template (Here person is the input variable)
   template="when was {person} born"  # Template for the prompt
)


# Chain of LLMs
chain2 = LLMChain(  # Second Chain Which uses the second prompt template to find the person's DOB
   llm=llm, prompt=second_input_prompt, verbose=True, output_key='dob'
)


# Prompt Templates
third_input_prompt = PromptTemplate( # Third Prompt Template
   input_variables=['dob'], # Input variables for creating the prompt template (Here dob is the input variable)
   template="Mention 5 major events happened around {dob} in the world" # Template for the prompt
)


chain3 = LLMChain(llm=llm, prompt=third_input_prompt, verbose=True, output_key='description') # Third Chain Which uses the third prompt template to find the person's DOB



# Additional Prompt Templates
fourth_input_prompt = PromptTemplate(
    input_variables=['person'],
    template="What is the nationality of {person}"
)

# Additional Chains
chain4 = LLMChain(
    llm=llm, prompt=fourth_input_prompt, verbose=True, output_key='nationality'
)


# Additional Prompt Templates
fifth_input_prompt = PromptTemplate(
    input_variables=['person'],
    template="What are some good habits of {person}"
)
chain5 = LLMChain(
    llm=llm, prompt=fifth_input_prompt, verbose=True, output_key='good_habits'
)
sixth_input_prompt = PromptTemplate(
    input_variables=['person'],
    template="What are some bad habits of {person}"
)
chain6 = LLMChain(
    llm=llm, prompt=sixth_input_prompt, verbose=True, output_key='habits'
)
seventh_input_prompt = PromptTemplate(
    input_variables=['person'],
    template="Describe the physical appearance of {person}"
)
chain7 = LLMChain(
    llm=llm, prompt=seventh_input_prompt, verbose=True, output_key='physical_appearance'
)
eighth_input_prompt = PromptTemplate(
    input_variables=['person'],
    template="What are the skills of {person}"
)
chain8 = LLMChain(
    llm=llm, prompt=eighth_input_prompt, verbose=True, output_key='skills'
)
ninth_input_prompt = PromptTemplate(
    input_variables=['person'],
    template="What is the educational background of {person}"
)
chain9 = LLMChain(
    llm=llm, prompt=ninth_input_prompt, verbose=True, output_key='education'
)


tenth_input_prompt = PromptTemplate(
    input_variables=['person'],
    template="What is the profession of {person}"
)
chain10 = LLMChain(
    llm=llm, prompt=tenth_input_prompt, verbose=True, output_key='profession'
)

eleventh_input_prompt = PromptTemplate(
    input_variables=['person'],
    template="What are recent controversies surrounding {person}"
)
chain11 = LLMChain(
    llm=llm, prompt=eleventh_input_prompt, verbose=True, output_key='controversy'
)

twelfth_input_prompt = PromptTemplate(
    input_variables=['person'],
    template="What are some notable awards or achievements of {person}"
)
chain12 = LLMChain(
    llm=llm, prompt=twelfth_input_prompt, verbose=True, output_key='awards'
)


thirteenth_input_prompt = PromptTemplate(
    input_variables=['person'],
    template="Tell me about {person}'s family background and mention name of relatives of {person} which are famous for good cause"
)
chain13 = LLMChain(
    llm=llm, prompt=thirteenth_input_prompt, verbose=True, output_key='family_background'
)


fourteenth_input_prompt = PromptTemplate(
    input_variables=['person'],
    template="what are official endorsement of {person}"
)
chain14 = LLMChain(
    llm=llm, prompt=fourteenth_input_prompt, verbose=True, output_key='social_media'
)
fifteenth_input_prompt = PromptTemplate(
    input_variables=['person'],
    template="tell me about the wardrobe and choice of wear of  {person}"
)
chain15 = LLMChain(
    llm=llm, prompt=fifteenth_input_prompt, verbose=True, output_key='wardrobe'  # Corrected typo here
)

sixteenth_input_prompt = PromptTemplate(
    input_variables=['person'],
    template="which superhero is favourite of {person}? in one or two word"
)
chain16 = LLMChain(
    llm=llm, prompt=sixteenth_input_prompt, verbose=True, output_key='superhero'
)
seventeenth_input_prompt = PromptTemplate(
    input_variables=['person'],
    template="what are the  project ,projects such as movies ,of {person} ? mention projects which are done , recent done and upcoming projects."
)
chain17 = LLMChain(
    llm=llm, prompt=seventeenth_input_prompt, verbose=True, output_key='projects'
)
# Parent Chain
parent_chain = SequentialChain(
    chains=[chain, chain2, chain3, chain4, chain5, chain6, chain7, chain8, chain9, chain10, chain11, chain12, chain13, chain14,chain15,chain16,chain17], 
    input_variables=['name'],
    output_variables=['person', 'dob', 'description', 'nationality', 'good_habits', 'habits', 'physical_appearance', 'skills', 'education', 'profession', 'controversy', 'awards', 'family_background', 'social_media','wardrobe','superhero','projects'],
    verbose=True
)
if input_text:
    result = parent_chain({'name': input_text})
  
    with st.expander("Name"):
        st.write(result['name'])

    # Creating columns for organizing the expanders side by side
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        with st.expander("Details"):
            st.write(result['person'])

        with st.expander("Date of Birth"):
            st.write(result['dob'])

        with st.expander("Nationality"):
            st.write(result['nationality'])

        with st.expander("Wardrobe"):
            st.write(result['wardrobe'])
         
        with st.expander("Favourite SuperHero"):
            st.write(result['superhero'])




    with col2:
        with st.expander("Around his DOM in the world"):
            st.write(result['description'])

        with st.expander("Good Habits"):
            st.write(result['good_habits'])

        with st.expander("Habits"):
            st.write(result['habits'])

        with st.expander("Physical Appearance"):
            st.write(result['physical_appearance'])

    
        with st.expander("Projects"):
            st.write(result['projects'])

    with col3:
        with st.expander("Controversies"):
            st.write(result['controversy'])

        with st.expander("Endorsements"):
            st.write(result['social_media'])

        with st.expander("Known Relation"):
            st.write(result['family_background'])

        with st.expander("Achievments"):
            st.write(result['awards'])
        

      
    st.write('Data Fetched Successfully')
