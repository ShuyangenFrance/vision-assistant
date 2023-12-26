import playsound
from gtts import gTTS
import speech_recognition as sr

from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, AgentType
import os
from tools import tools
import logging

openai_key=os.environ["OPENAI_API_KEY"]

def main():
    conversational_memory = ConversationBufferMemory(
        memory_key='chat_history',
        k=5,
        return_messages=True
    )

    llm = ChatOpenAI(
        openai_api_key=openai_key,
        temperature=0,
        model_name="gpt-3.5-turbo"
    )

    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        memory=conversational_memory,
        verbose=True
    )

    recognizer = sr.Recognizer()

    while True:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source,duration=4)
            while True:
                playsound.playsound("greeting_voice.mp3")
                audio=recognizer.listen(source)
                try:
                    user_input=recognizer.recognize_google(audio, language='en')
                    logging.info(f"user input: {user_input}")
                    response = agent.run(f"{user_input}. "
                                         f"The image path is YOUR IMAGE PATH")
                    logging.info(f"response: {response}")
                    response_voice = gTTS(text=response, lang='en', slow=False)
                    response_voice.save("response_voice.mp3")
                    playsound.playsound("response_voice.mp3")
                    os.remove("response_voice.mp3")
                except Exception:
                    continue

if __name__=="__main__":
    main()
