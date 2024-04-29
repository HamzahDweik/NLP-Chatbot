import re
import datetime
import wikipedia
import random
from collections import deque
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import pickle
from os import path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sys
# from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QTextEdit, QLineEdit, QPushButton, QLabel, QInputDialog
# from PyQt5.QtCore import pyqtSlot
# from PyQt5.QtGui import QFont
from utils import *
from gensim.models import KeyedVectors

embeddings_model = KeyedVectors.load_word2vec_format('saves\GoogleNews-vectors-negative300.bin', binary=True)
minecraft_guru, tokenizer_ques, tokenizer_ans = load_guru()
def generate_text(input): return generate_text_(input, minecraft_guru, tokenizer_ques, tokenizer_ans)

CONTEXT_LENGTH = 3

class UserModel:
    """
    Represents a user model to store user-specific information including sentiment and memory.

    Attributes:
        name (str): The name of the user.
        sentiment (dict): A dictionary to store sentiments related to various topics.
    """

    def __init__(self, name: str):
        self.name = name
        self.sentiment = []

    def update_sentiment(self, text):
        """Adds or updates sentiment for a given topic."""
        if ' like ' in text:
            self.sentiment.append(text)


    def save_user(self):
        """Saves the current state of the user model."""
        return {'name': self.name, 'sentiment': self.sentiment, 'memory': self.memory}


# class ChatbotGUI(QWidget):
#     """
#     A graphical user interface for the chatbot using PyQt5.

#     The GUI includes a text area for conversation display, a text input for user input, and a send button.
#     """

#     def __init__(self):
#         """
#         Initializes the GUI components and layout.
#         """
#         super().__init__()
#         self.initUI()
#         self.askUserName()

#     def initUI(self):
#         """
#         Sets up the GUI layout and components, including styling.
#         """
#         layout = QVBoxLayout()
#         layout.setSpacing(15)  # increase spacing for better readability

#         # conversation display area
#         self.conversationArea = QTextEdit()
#         self.conversationArea.setReadOnly(True)
#         self.conversationArea.setFont(QFont('Arial', 12))
#         self.conversationArea.setStyleSheet(
#             "background-color: #f2f2f2; color: #333333; border: 1px solid #d9d9d9; border-radius: 8px; padding: 10px;")
#         layout.addWidget(self.conversationArea)

#         # user input field
#         self.userInput = QLineEdit(self)
#         self.userInput.setFont(QFont('Arial', 12))
#         self.userInput.setStyleSheet(
#             "background-color: #ffffff; color: #333333; border: 1px solid #cccccc; border-radius: 8px; padding: 5px;")
#         layout.addWidget(self.userInput)

#         # send button
#         self.sendButton = QPushButton('Send')
#         self.sendButton.setFont(QFont('Arial', 12))
#         self.sendButton.setStyleSheet(
#             "QPushButton { background-color: #007bff; color: white; border-radius: 8px; padding: 6px 12px; }"
#             "QPushButton:hover { background-color: #0069d9; }"
#             "QPushButton:pressed { background-color: #0056b3; }")
#         layout.addWidget(self.sendButton)

#         self.setLayout(layout)
#         self.setWindowTitle('Chatbot GUI')
#         self.setGeometry(300, 300, 600, 500)
#         self.sendButton.clicked.connect(self.on_send)

#     def askUserName(self):
#         """
#         Prompts the user for their name and creates a user entity.
#         """
#         name, okPressed = QInputDialog.getText(self, greeting(), "Your name:", QLineEdit.Normal, "")
#         if okPressed and name != '':
#             if name in users:
#                 user = users[name]
#                 self.display_message("HollandBot", "Welcome back, " + name + "! " + introduction())
#                 users[user.name] = user
#             else:
#                 users[name] = UserModel(name=name)
#                 user = users[name]
#                 self.display_message("HollandBot", "It's nice to meet you, " + name + "! " + introduction())
#             self.user = user
#         else:
#             self.close()  # Close the application if no name is provided
        

#     @pyqtSlot()
#     def on_send(self):
#         """
#         Slot for handling the send button click event.
#         """
#         user_message = self.userInput.text()
#         self.userInput.clear()
#         response = generate_response(user_message) + " " + ask_for_input(self.user)
#         self.display_message("You", user_message)
#         self.display_message("HollandBot", response)

#     def display_message(self, sender, message):
#         """
#         Displays a formatted message in the conversation area.

#         Args:
#             sender (str): The sender of the message.
#             message (str): The message content.
#         """
#         formatted_message = f"<b>{sender}:</b> {message}<br/>"
#         self.conversationArea.append(formatted_message)

#     def closeEvent(self, event):
#         """
#         Event handler called when the window is closed.
#         """
#         dump_users(users)
#         event.accept()  # proceed with the window closure

def dump_users(users, filename='saves/users.pkl'):
    """
    Dumps the user data to a file using pickle.

    Args:
        users (dict): The dictionary containing user data.
        filename (str): The name of the file to dump data into.
    """
    with open(filename, 'wb') as file:
        pickle.dump(users, file)


def initialize_users():
    """
    Initializes and saves an empty user dictionary.
    """
    users = dict()
    with open('saves/users.pkl', 'wb') as file:
        pickle.dump(users, file, protocol=pickle.HIGHEST_PROTOCOL)

def load_user_data():
    """
    Loads user data from a pickle file, or initializes if it doesn't exist.

    Returns:
        dict: A dictionary of user data.
    """
    if not path.exists('saves/users.pkl'): 
        initialize_users()
    with open('saves/users.pkl', 'rb') as file:
        return pickle.load(file)

def preprocess_text(text):
    """
    Preprocesses text by tokenizing, removing stopwords, and lemmatizing.

    Args:
        text (str): Text to preprocess.

    Returns:
        list: A list of preprocessed tokens.
    """
    tokens = word_tokenize(text.lower())   
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if w not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(t) for t in tokens] 
    return lemmatized

def calculate_similarity(input_tokens, choices):
    """
    Calculates the cosine similarity between the input tokens and a list of choices.

    Args:
        input_tokens (list): A list of preprocessed input tokens.
        choices (list): A list of strings to compare against the input tokens.

    Returns:
        tuple: A tuple containing the best match from choices and its similarity score.
    """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(choices + [' '.join(input_tokens)]) 
    input_vector = vectorizer.transform([' '.join(input_tokens)]) 
    scores = [cosine_similarity(input_vector.reshape(1, -1), tfidf_matrix[i].reshape(1, -1))[0][0] for i in range(len(choices))]
    best_choice_index = scores.index(max(scores))
    return choices[best_choice_index], scores[best_choice_index]

def respond_to_self_inquiry(user_input):
    # List of phrases that might mean the user is asking about the chatbot
    inquiry_keywords = [
        "who are you", "what is your name", "who am I speaking to",
        "tell me about yourself", "what are you", "are you a robot",
        "are you human", "what can you do", "how were you made",
        "who created you", "where do you come from", "what is your purpose",
        "how do you work", "can you help me", "what is your function",
        "are you intelligent", "do you understand me", "how smart are you",
        "what do you know", "are you a chatbot", "do you have feelings",
        "how old are you"
    ]

    # Responses the chatbot can choose from
    responses = [
        "Hello! I'm a chatbot here to help you with any questions you have about Minecraft.",
        "I'm your friendly neighborhood chatbot, designed to assist you with Minecraft queries.",
        "Greetings! I'm an AI developed to provide guidance and answers about Minecraft.",
        "Hi there! Think of me as your assistant for all things Minecraft.",
        "I'm an artificial intelligence programmed to help you understand and enjoy Minecraft better.",
        "Hello! I'm crafted by humans to assist in your Minecraft adventure.",
        "I'm a virtual helper here to navigate the world of Minecraft with you.",
        "Call me your Minecraft guide, ready to assist with any information you need.",
        "Hi! I'm here to make your experience with Minecraft smoother and more fun.",
        "As a chatbot, I'm programmed to provide answers and support for your Minecraft queries."
    ]

    # Check if the input is about the chatbot itself
    if any(term in user_input.lower() for term in inquiry_keywords):
        return random.choice(responses)
    else:
        return None
def check_for_fact(user_input):
    # Check if user asks for a random fact
    if " fact " in user_input.lower():
        random_questions = [
            "What is the level range where diamond ore can be found in Minecraft?",
            "What is the chance of obtaining a specific mineral when brushing suspicious sand in desert pyramids?",
            "What is the primary use of diamonds obtained from suspicious sand loot?",
            "What item can expert-level armorer, toolsmith, and weaponsmith villagers in Bedrock Edition trade for an emerald?",
            "What is the probability of an expert-level toolsmith villager offering to buy a diamond for an emerald in the Java Edition of Minecraft?",
            "How do you get a diamond in Minecraft?",
            "Can I use a diamond ingot as a substitute for a certain type of ingot in beacons?",
            "What is the most efficient way to obtain a large quantity of coal in the Nether, considering the rarity of coal ore and the limited inventory space in the Nether Fortress?",
            "What type of data value is related to 'Diamond' in the Bedrock Edition of Minecraft and where can issues regarding it be reported?",
            "What is the most common cause of naturally occurring diamonds in Minecraft?",
            "What type of ore is typically found near lava in a Minecraft gallery?",
            "What type of merchandise does JINX create for Minecraft, and is there any notable artwork featuring diamonds?",
            "Can any tool be used to instantly break an element in Minecraft Education?",
            "What is the primary method for obtaining elements and isotopes in Minecraft, according to the element constructor?",
            "How can you obtain the Unknown element in Minecraft without using the crafting table?",
            "What is a possible way to obtain a material reducer in Survival mode without using commands?",
            "What is the most common element in the Earth's crust?",
            "What is the most abundant isotope of lithium, an alkali metal?",
            "What is the most abundant alkaline earth metal in the Earth's crust, and what is its most common isotope?",
            "What is the most abundant post-transition metal in the Earth's crust, and which of its isotopes is most commonly used as a calibration standard in mass spectrometry?",
            "What is the most abundant non-metal isotope of carbon, and what is its atomic mass?",
            "What is the atomic number of astatine, a halogen isotope that is a radioactive decay product of uranium and thorium, and has a half-life of 8.1 hours?",
            "What is the name of the texture used in the top of the Calibrated Sculk Sensor in Minecraft?",
            "What is the name of the block that is used to charge your devices?",
            "What is the purpose of the Rainbow Wool side a, side b, side c, side d, side e, and side f in Minecraft Earth?",
            "What is the version number of the Bedrock Edition released for PlayStation consoles on December 2, 2021?",
            "What is the term used for the Xbox 360 Edition of Minecraft to describe a major update?",
            "What is the fastest way to mine pumpkins in Minecraft?",
            "Why do pumpkins break when pushed by a piston, but not when pulled?",
            "What type of villages can you find pumpkins naturally generating in pile form?",
            "What is the key difference between pumpkin farming and melon farming in Minecraft?",
            "Can I speed up the growth of pumpkin stems by using bone meal on them, and if so, will it immediately produce a pumpkin?",
            "How many pumpkins do apprentice-level farmer villagers buy from other players in Bedrock Edition?",
            "What is the probability of a apprentice-level farmer villager buying 6 pumpkins for one emerald?",
            "What is the use of pumpkins in Minecraft?",
            "What is the purpose of using note blocks in a Minecraft world?",
            "What is the primary difference between the Bedrock Edition and the Java Edition of Minecraft in terms of sound generation?",
            "What is the unique sound effect used in the Bedrock Edition of Minecraft to indicate the presence of a player's bed, which is different from the sound effect used in the Java Edition?",
            "What is the default material for crafting a stone axe?",
            "What is the specific tracker where issues related to pumpkins in the Bedrock Edition of Minecraft are maintained?",
            "What block states are affected when a group of pumpkins is placed?",
            "What type of biome can naturally spawn pumpkins?",
            "What is the type of pickaxe required to mine a quartz pillar in Minecraft?",
            "What do master-level mason villagers sell in Minecraft for an emerald?",
            "How can you efficiently trade quartz pillars in Minecraft while taking into account the constraints of their rotation?",
            "How can I create a unique sound effect in Minecraft by using quartz pillars and note blocks?"
        ]
        return generate_text(random.choice(random_questions))
    else: return None



def compute_embeddings(text):
    # Tokenize the text and filter out words not in the model's vocabulary
    words = text.lower().split()
    words = [word for word in words if word in embeddings_model.key_to_index]

    # If no words from the text are in the model's vocabulary, return a zero vector
    if not words:
        return np.zeros(embeddings_model.vector_size)

    # Compute the average of the embeddings of the words
    word_embeddings = [embeddings_model[word] for word in words]
    sentence_embedding = np.mean(word_embeddings, axis=0)
    return sentence_embedding

def cosine_similarity(vec1, vec2):
    """Compute the cosine similarity between two vectors."""
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        # Avoid division by zero
        return 0.0
    similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return similarity

def not_sure():
    responses = [
        "I'm sorry, I couldn't provide a useful answer to that. Could you rephrase the question more clearly?",
        "Apologies, but I'm not able to provide a helpful response right now. Could you clarify your question?",
        "I'm unsure how to answer that effectively. Can you phrase it differently?",
        "Sorry, I couldn't quite catch that. Can you restate your question more clearly?",
        "My apologies, I'm unable to give a useful answer. Could you rephrase your inquiry?",
        "Unfortunately, I can't provide a useful response at the moment. Can you reword your question?",
        "I'm having trouble understanding that. Can you try asking in a different way?",
        "Sorry, I'm not able to answer that effectively. Could you clarify what you're asking?",
        "I'm sorry, that's unclear to me. Could you phrase your question differently?",
        "Apologies, but I couldn't give a helpful answer there. Can you ask in another way?"
    ]
    return random.choice(responses)

def generate_response(user_input):
    """
    Generates a response to the user input by processing the input and finding the best match from the knowledge base.

    Args:
        user_input (str): The user's input text.

    Returns:
        str: A response generated based on the user input.
    """

    # preprocessing of the input
    modified_input = re.sub(r"[^a-zA-Z0-9? ]", "", user_input.lower())

    # checks for self-inquiry style questions
    self_inquiry = respond_to_self_inquiry(modified_input)
    if self_inquiry: return self_inquiry

    # checks for random facts inquiry
    random_fact = check_for_fact(modified_input)
    if random_fact: return random_fact

    # generates a response to the question
    potential_response = generate_text(modified_input)
    response_embedding = compute_embeddings(potential_response)
    question_embedding = compute_embeddings(modified_input)

    similarity_score = cosine_similarity(question_embedding, response_embedding)
    if similarity_score > 0.5: return potential_response
    else: return not_sure()
    

def introduction():
    """
    Generates an introduction message that explains the bot's expertise in Tom Holland and asks the user about their questions.

    Returns:
        str: An introduction message.
    """

    # List of introduction choices
    intro_choices = [
        "I'm a bot with a wealth of knowledge about Tom Holland. ",
        "I specialize in all things Tom Holland. ",
        "I'm your go-to source for any Tom Holland queries. ",
        "If you've got questions about Tom Holland, you're in the right place. ",
        "Tom Holland facts and information are my forte. "
    ]
    intro_choice = random.choice(intro_choices)

    # List of assist choices
    assist_choices = [
        "What would you like to know about him?",
        "Feel free to ask any question about Tom Holland.",
        "I'm here to answer all your Tom Holland-related questions.",
        "Do you have any queries regarding Tom Holland?",
        "Let me know what you're curious about in relation to Tom Holland."
    ]
    assist_choice = random.choice(assist_choices)

    return intro_choice + assist_choice

def greeting():
    """
    Generates a greeting message based on the current time.

    Returns:
        str: A greeting message.
    """
    
    # list of greeting choices
    choices = ['Hello, ', 'Greetings! ', 'Hey, ', "How's it going?", 'time-based', 'time-based', 'time-based']
    choice = random.choice(choices)

    # determine greeting based on the current time
    if choice == 'time-based':
        current_hour = datetime.datetime.now().hour
        if 0 <= current_hour < 12:
            choice = "Good morning!"
        elif 12 <= current_hour < 17:
            choice = "Good afternoon!"
        else:
            choice = "Good evening!"
    
    # Add a follow-up question to the greeting
    choices = ["Who do I have the pleasure of speaking with today?", "What's your name?", 'To whom am I speaking?']
    second_choice = random.choice(choices)
    return choice + " I am the minecraft guru! " + second_choice + " "

def greet_user(users):
    """
    Greets the user and retrieves or creates a UserModel instance.

    Args:
        users (dict): A dictionary of existing user models.

    Returns:
        UserModel: The user model for the current user.
    """
    name = input(greeting())
    lower_name = name.lower()

    # create a new user or retrieve existing user data
    if lower_name in users:
        user = users[lower_name]
        print("Welcome back " + user.name + "!")
    else:
        users[lower_name] = UserModel(name=name)
        user = users[lower_name]
        print("It's nice to meet you " + user.name)
    
    return user

def search_wikipedia(query):
    """
    Searches Wikipedia for a summary of a given query.

    Args:
        query (str): The Wikipedia query.

    Returns:
        str: A summary of the Wikipedia article or an error message.
    """
    try:
        result = wikipedia.summary(query, sentences=5)
        return f"Here's some information about {query}: {result}"
    except Exception as e: 
        return "Hmm, something went wrong while searching for this movie. Perhaps try again with another one?"        
    
def ask_for_input(user):
    """
    Asks the user for input with a dynamic prompt.

    Args:
        user (UserModel): The current user model.

    Returns:
        str: A question for the user to get input.
    """
    prompts = [
        "Do you have any other questions?",
        f"Is there anything else you're curious about, {user.name}?",
        "What else would you like to know?",
        f"{user.name}, do you have a follow-up question or another topic in mind?"
    ]

    user_response = input(random.choice(prompts) + f'\n{user.name}: ')
    return user_response


def chat():
    """
    Initiates and manages the chat session with the user.

    Uses the user model and user input to generate and display responses.
    """
    users = load_user_data()
    user = greet_user(users)

    while True:
        user_input = ask_for_input(user)
        if user_input == 'q': dump_users(users); break
        response = generate_response(user_input) 
        print(f'MinecraftGuru: {response}', end=' ')
        
    # load GUI
    # app = QApplication(sys.argv)
    # ex = ChatbotGUI()
    # ex.show()
    # sys.exit(app.exec_())
    # chat()
