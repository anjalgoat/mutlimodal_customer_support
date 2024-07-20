from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

def detect_intent_with_llm(user_query: str) -> str:
    
    template = """
    You are an AI trained to detect the intent behind user queries. Below are some common intents:
    Just reply with a intention.
    Dont explain the reason


    - new_account: Queries related to opening or creating a new account.
    - balance_check: Queries related to checking account balance.
    - loan_inquiry: Queries related to applying for a loan or loan details.
    - lost_card: Queries related to lost or stolen cards.
    - transaction_issue: Queries related to transaction problems.
    If nothing matches this then
    simply reply N/A
    Based on the user's query, determine the intent.

    User Query: {user_query}
    Detected Intent:
    """

    formatted_prompt = template.format(user_query=user_query)

    
    llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0)

    
    response = llm.invoke(formatted_prompt)
    detected_intent = response.content.strip()

    return detected_intent