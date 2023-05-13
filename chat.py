# # # # import random
# # # # import json
# # # #
# # # # import torch
# # # #
# # # # from model import NeuralNet
# # # # from nltk_utils import bag_of_words, tokenize
# # # # from db_sqlite import respond
# # # #
# # # # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# # # #
# # # # with open('intents.json', 'r') as json_data:
# # # #     intents = json.load(json_data)
# # # #
# # # # FILE = "data.pth"
# # # # data = torch.load(FILE)
# # # #
# # # # input_size = data["input_size"]
# # # # hidden_size = data["hidden_size"]
# # # # output_size = data["output_size"]
# # # # all_words = data['all_words']
# # # # tags = data['tags']
# # # # model_state = data["model_state"]
# # # #
# # # # model = NeuralNet(input_size, hidden_size, output_size).to(device)
# # # # model.load_state_dict(model_state)
# # # # model.eval()
# # # #
# # # # bot_name = "Sam"
# # # # print("Let's chat! (type 'quit' to exit)")
# # # # while True:
# # # #     # sentence = "do you use credit cards?"
# # # #     sentence = input("You: ")
# # # #     if sentence == "quit":
# # # #         break
# # # #
# # # #     sentence = tokenize(sentence)
# # # #     X = bag_of_words(sentence, all_words)
# # # #     X = X.reshape(1, X.shape[0])
# # # #     X = torch.from_numpy(X).to(device)
# # # #
# # # #     output = model(X)
# # # #     _, predicted = torch.max(output, dim=1)
# # # #
# # # #     tag = tags[predicted.item()]
# # # #
# # # #     probs = torch.softmax(output, dim=1)
# # # #     prob = probs[0][predicted.item()]
# # # #     if prob.item() > 0.75:
# # # #         for intent in intents['intents']:
# # # #             if tag == intent["tag"]:
# # # #                 print(f"{bot_name}: {random.choice(intent['responses'])}")
# # # #     else:
# # # #         query = f"SELECT drinks FROM menu"
# # # #         response = respond(query)
# # # #         if response:
# # # #             print("Coffee shop assistant: We have " + response + " on our menu.")
# # # #         else:
# # # #             print("Coffee shop assistant: I'm sorry, I don't understand. Can you please try again?")
# # # #
# # # # import random
# # # # import json
# # # # import sqlite3
# # # #
# # # # conn = sqlite3.connect('menu.db')
# # # # c = conn.cursor()
# # # # import torch
# # # #
# # # # from model import NeuralNet
# # # # from nltk_utils import bag_of_words, tokenize
# # # #
# # # # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# # # #
# # # # with open('intents.json', 'r') as json_data:
# # # #     intents = json.load(json_data)
# # # #
# # # # FILE = "data.pth"
# # # # data = torch.load(FILE)
# # # #
# # # # input_size = data["input_size"]
# # # # hidden_size = data["hidden_size"]
# # # # output_size = data["output_size"]
# # # # all_words = data['all_words']
# # # # tags = data['tags']
# # # # model_state = data["model_state"]
# # # #
# # # # model = NeuralNet(input_size, hidden_size, output_size).to(device)
# # # # model.load_state_dict(model_state)
# # # # model.eval()
# # # #
# # # # bot_name = "Sam"
# # # # print("Let's chat! (type 'quit' to exit)")
# # # # while True:
# # # #     # sentence = "do you use credit cards?"
# # # #     sentence = input("You: ")
# # # #     if sentence == "quit":
# # # #         break
# # # #
# # # #     sentence = tokenize(sentence)
# # # #     X = bag_of_words(sentence, all_words)
# # # #     X = X.reshape(1, X.shape[0])
# # # #     X = torch.from_numpy(X).to(device)
# # # #
# # # #     output = model(X)
# # # #     _, predicted = torch.max(output, dim=1)
# # # #
# # # #     tag = tags[predicted.item()]
# # # #
# # # #     probs = torch.softmax(output, dim=1)
# # # #     prob = probs[0][predicted.item()]
# # # #     if prob.item() > 0.75:
# # # #         for intent in intents['intents']:
# # # #             if tag == intent["tag"]:
# # # #                 # Check if the user asked for drinks or prices
# # # #                 if intent['tag'] == 'drinks':
# # # #                     # Execute a query to retrieve the drinks from the menu table
# # # #                     c.execute("SELECT name FROM menu")
# # # #                     drinks = c.fetchall()
# # # #                     drinks_list = [drink[0] for drink in drinks]
# # # #                     response = f"We have {', '.join(drinks_list)}"
# # # #
# # # #                 elif intent['tag'] == 'prices':
# # # #                     # Execute a query to retrieve the prices from the menu table
# # # #                     c.execute("SELECT name, price FROM menu")
# # # #                     menu_items = c.fetchall()
# # # #                     response = "Here are our prices:\n"
# # # #                     for item in menu_items:
# # # #                         response += f"{item[0]} - ${item[1]}\n"
# # # #
# # # #                 else:
# # # #                     response = random.choice(intent['responses'])
# # # #
# # # #                 print(f"{bot_name}: {response}")
# # # #     else:
# # # #         print(f"{bot_name}: I do not understand...")
# # # # import random
# # # # import json
# # # #
# # # # import torch
# # # # import sqlite3
# # # #
# # # # from model import NeuralNet
# # # # from nltk_utils import bag_of_words, tokenize
# # # #
# # # # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# # # #
# # # # with open('intents.json', 'r') as json_data:
# # # #     intents = json.load(json_data)
# # # #
# # # # FILE = "data.pth"
# # # # data = torch.load(FILE)
# # # #
# # # # input_size = data["input_size"]
# # # # hidden_size = data["hidden_size"]
# # # # output_size = data["output_size"]
# # # # all_words = data['all_words']
# # # # tags = data['tags']
# # # # model_state = data["model_state"]
# # # #
# # # # model = NeuralNet(input_size, hidden_size, output_size).to(device)
# # # # model.load_state_dict(model_state)
# # # # model.eval()
# # # #
# # # # bot_name = "Sam"
# # # # print("Let's chat! (type 'quit' to exit)")
# # # #
# # # # # Connect to the database
# # # # conn = sqlite3.connect('menu.db')
# # # #
# # # # while True:
# # # #     # sentence = "do you use credit cards?"
# # # #     sentence = input("You: ")
# # # #     if sentence == "quit":
# # # #         break
# # # #
# # # #     sentence = tokenize(sentence)
# # # #     X = bag_of_words(sentence, all_words)
# # # #     X = X.reshape(1, X.shape[0])
# # # #     X = torch.from_numpy(X).to(device)
# # # #
# # # #     output = model(X)
# # # #     _, predicted = torch.max(output, dim=1)
# # # #
# # # #     tag = tags[predicted.item()]
# # # #
# # # #     probs = torch.softmax(output, dim=1)
# # # #     prob = probs[0][predicted.item()]
# # # #     if prob.item() > 0.75:
# # # #         for intent in intents['intents']:
# # # #             if tag == intent["tag"]:
# # # #                 if intent['tag'] == 'drinks':
# # # #                     # Retrieve the list of drinks from the database
# # # #                     cursor = conn.execute("SELECT name FROM drinks")
# # # #                     drinks_list = [row[0] for row in cursor.fetchall()]
# # # #                     response = ', '.join(drinks_list)
# # # #                     print(f"{bot_name}: We have {response}")
# # # #                 elif intent['tag'] == 'prices':
# # # #                     # Retrieve the list of prices from the database
# # # #                     cursor = conn.execute("SELECT price FROM drinks")
# # # #                     prices_list = [row[0] for row in cursor.fetchall()]
# # # #                     response = ', '.join([f"${price:.2f}" for price in prices_list])
# # # #                     print(f"{bot_name}: Our prices are {response}")
# # # #                 else:
# # # #                     print(f"{bot_name}: {random.choice(intent['responses'])}")
# # # #     else:
# # # #         print(f"{bot_name}: I do not understand...")
# # # #
# # # # # Close the database connection
# # # # conn.close()
# # # from db_sqlite import get_drinks, get_price
# # # import random
# # # import json
# # #
# # # import torch
# # #
# # # from model import NeuralNet
# # # from nltk_utils import bag_of_words, tokenize
# # #
# # # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# # #
# # # with open('intents.json', 'r') as json_data:
# # #     intents = json.load(json_data)
# # #
# # # FILE = "data.pth"
# # # data = torch.load(FILE)
# # #
# # # input_size = data["input_size"]
# # # hidden_size = data["hidden_size"]
# # # output_size = data["output_size"]
# # # all_words = data['all_words']
# # # tags = data['tags']
# # # model_state = data["model_state"]
# # #
# # # model = NeuralNet(input_size, hidden_size, output_size).to(device)
# # # model.load_state_dict(model_state)
# # # model.eval()
# # #
# # # bot_name = "Sam"
# # # print("Let's chat! (type 'quit' to exit)")
# # # while True:
# # #     # sentence = "do you use credit cards?"
# # #     sentence = input("You: ")
# # #     if sentence == "quit":
# # #         break
# # #
# # #     sentence = tokenize(sentence)
# # #     X = bag_of_words(sentence, all_words)
# # #     X = X.reshape(1, X.shape[0])
# # #     X = torch.from_numpy(X).to(device)
# # #
# # #     output = model(X)
# # #     _, predicted = torch.max(output, dim=1)
# # #
# # #     tag = tags[predicted.item()]
# # #
# # #     probs = torch.softmax(output, dim=1)
# # #     prob = probs[0][predicted.item()]
# # #
# # #     if sentence == 'what drinks do you have?':
# # #         drinks = get_drinks()
# # #         print("We have the following drinks: " + ', '.join(drinks))
# # #     elif ' '.join(sentence).startswith('how much is'):
# # #         drink = sentence[10:].strip()
# # #         price = get_price(drink)
# # #         if price:
# # #             print(f"The price of {drink} is {price} dollars.")
# # #         else:
# # #             print(f"Sorry, we don't have {drink} in our menu.")
# # #     elif prob.item() > 0.75:
# # #         for intent in intents['intents']:
# # #             if tag == intent["tag"]:
# # #                 print(f"{bot_name}: {random.choice(intent['responses'])}")
# # #     else:
# # #         print(f"{bot_name}: I do not understand...")
# from db_sqlite import get_drinks, get_price
# import random
# # import json
# # import sqlite3
# #
# # import torch
# #
# # from model import NeuralNet
# # from nltk_utils import bag_of_words, tokenize
# #
# # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# #
# # with open('intents.json', 'r') as json_data:
# #     intents = json.load(json_data)
# #
# # FILE = "data.pth"
# # data = torch.load(FILE)
# #
# # input_size = data["input_size"]
# # hidden_size = data["hidden_size"]
# # output_size = data["output_size"]
# # all_words = data['all_words']
# # tags = data['tags']
# # model_state = data["model_state"]
# #
# # model = NeuralNet(input_size, hidden_size, output_size).to(device)
# # model.load_state_dict(model_state)
# # model.eval()
# #
# # conn = sqlite3.connect('menu.db')
# # c = conn.cursor()
# #
# # def get_response_from_database(query):
# #     conn = sqlite3.connect('menu.db')
# #     c = conn.cursor()
# #     c.execute("SELECT price FROM drinks WHERE name=?", (query,))
# #     response = c.fetchone()
# #     conn.close()
# #     if response:
# #         return response[0]
# #     else:
# #         return None
# #
# # bot_name = "Sam"
# #print("Let's chat! (type 'quit' to exit)")
# # while True:
# #     # sentence = "do you use credit cards?"
# #     sentence = input("You: ")
# #     if sentence == "quit":
# #         break
# #
# #     sentence = tokenize(sentence)
# #     X = bag_of_words(sentence, all_words)
# #     X = X.reshape(1, X.shape[0])
# #     X = torch.from_numpy(X).to(device)
# #
# #     output = model(X)
# #     _, predicted = torch.max(output, dim=1)
# #
# #     tag = tags[predicted.item()]
# #
# #     probs = torch.softmax(output, dim=1)
# #     prob = probs[0][predicted.item()]
# #
# #     if prob.item() > 0.75:
# #         if tag == 'prices':
# #             item = ' '.join(sentence)
# #             price = get_price(item)
# #             if price:
# #                 print(f"{bot_name}: The price of {item} is {price}.")
# #             else:
# #                 print(f"{bot_name}: Sorry, we don't have {item} in our shop.")
# #         else:
# #             for intent in intents['intents']:
# #                 if tag == intent["tag"]:
# #                     print(f"{bot_name}: {random.choice(intent['responses'])}")
# #     else:
# #         print(f"{bot_name}: I do not understand...")
# #
# # def get_price(item):
# #     c.execute(f"SELECT price FROM drinks WHERE name='{item}'")
# #     result = c.fetchone()
# #     if result:
# #         return result[0]
# #     else:
# #         return None
#
# #conn.close()
# # response = get_response_from_database(sentence)
# # if response:
# #     print(f"{bot_name}: {response}")
# # else:
# #     sentence = tokenize(sentence)
# #     X = bag_of_words(sentence, all_words)
# #     X = X.reshape(1, X.shape[0])
# #     X = torch.from_numpy(X).to(device)
# #
# #     output = model(X)
# #     _, predicted = torch.max(output, dim=1)
# #
# #     tag = tags[predicted.item()]
# #
# #     probs = torch.softmax(output, dim=1)
# #     prob = probs[0][predicted.item()]
# #     if prob.item() > 0.75:
# #         for intent in intents['intents']:
# #             if tag == intent["tag"]:
# #                 print(f"{bot_name}: {random.choice(intent['responses'])}")
# #     else:
# #         print(f"{bot_name}: I do not understand...")
#
# import random
# import json
# import sqlite3
#
# import torch
#
# from model import NeuralNet
# from nltk_utils import bag_of_words, tokenize
#
# # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# #
# # with open('intents.json', 'r') as json_data:
# #     intents = json.load(json_data)
# #
# # FILE = "data.pth"
# # data = torch.load(FILE)
# #
# # input_size = data["input_size"]
# # hidden_size = data["hidden_size"]
# # output_size = data["output_size"]
# # all_words = data['all_words']
# # tags = data['tags']
# # model_state = data["model_state"]
# #
# # model = NeuralNet(input_size, hidden_size, output_size).to(device)
# # model.load_state_dict(model_state)
# # model.eval()
# #
# # bot_name = "Sam"
# # print("Let's chat! (type 'quit' to exit)")
# # while True:
# #     # sentence = "do you use credit cards?"
# #     sentence = input("You: ")
# #     if sentence == "quit":
# #         break
# #
# #     sentence = tokenize(sentence)
# #     X = bag_of_words(sentence, all_words)
# #     X = X.reshape(1, X.shape[0])
# #     X = torch.from_numpy(X).to(device)
# #
# #     output = model(X)
# #     _, predicted = torch.max(output, dim=1)
# #
# #     tag = tags[predicted.item()]
# #
# #     probs = torch.softmax(output, dim=1)
# #     prob = probs[0][predicted.item()]
# #     if prob.item() > 0.75:
# #         for intent in intents['intents']:
# #             if tag == intent["tag"]:
# #                 if intent["tag"] == "how much is":
# #                     conn = sqlite3.connect('menu.db')
# #                     cursor = conn.cursor()
# #                     cursor.execute("SELECT price FROM drinks WHERE name=?", (sentence[-1],))
# #                     result = cursor.fetchone()
# #                     if result:
# #                         price = result[0]
# #                         print(f"{bot_name}: The price of {sentence[-1]} is {price}")
# #                     else:
# #                         print(f"{bot_name}: I'm sorry, I don't have that item.")
# #                     conn.close()
# #                 else:
# #                     print(f"{bot_name}: {random.choice(intent['responses'])}")
# #     else:
# #         print(f"{bot_name}: I do not understand...")
#
#
# # import random
# # import json
# # import torch
# # import sqlite3
# #
# # from model import NeuralNet
# # from nltk_utils import bag_of_words, tokenize
# #
# # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# #
# # with open('intents.json', 'r') as json_data:
# #     intents = json.load(json_data)
# #
# # FILE = "data.pth"
# # data = torch.load(FILE)
# #
# # input_size = data["input_size"]
# # hidden_size = data["hidden_size"]
# # output_size = data["output_size"]
# # all_words = data['all_words']
# # tags = data['tags']
# # model_state = data["model_state"]
# #
# # model = NeuralNet(input_size, hidden_size, output_size).to(device)
# # model.load_state_dict(model_state)
# # model.eval()
# #
# # bot_name = "Sam"
# # print("Let's chat! (type 'quit' to exit)")
# #
# # conn = sqlite3.connect('database.db')
# # cursor = conn.cursor()
# #
# # def get_answer(tag):
# #     cursor.execute("SELECT price FROM drinks WHERE name=?", (tag,))
# #     result = cursor.fetchone()
# #     if result:
# #         return result[0]
# #     else:
# #         return None
# #
# # while True:
# #     sentence = input("You: ")
# #     if sentence == "quit":
# #         break
# #
# #     # Check if the user has asked for the price of an item
# #     if "price of" in sentence:
# #         # Get the item name from the sentence
# #         item = sentence.split("price of ")[1]
# #         # Query the database for the price of the item
# #         cursor.execute("SELECT price FROM drinks WHERE name=?", (item,))
# #         result = cursor.fetchone()
# #         if result is None:
# #             # If the item is not found in the database, respond accordingly
# #             print(f"{bot_name}: I'm sorry, I don't know the price of {item}.")
# #         else:
# #             # If the item is found in the database, give the price
# #             price = result[0]
# #             print(f"{bot_name}: The price of {item} is {price}.")
# #     else:
# #         # If the user has not asked for the price of an item, respond with a generic message
# #         print(f"{bot_name}: I'm sorry, I don't understand.")
# #
# #
# #     sentence = tokenize(sentence)
# #     X = bag_of_words(sentence, all_words)
# #     X = X.reshape(1, X.shape[0])
# #     X = torch.from_numpy(X).to(device)
# #
# #     output = model(X)
# #     _, predicted = torch.max(output, dim=1)
# #
# #     tag = tags[predicted.item()]
# #
# #     probs = torch.softmax(output, dim=1)
# #     prob = probs[0][predicted.item()]
# #
# #     if prob.item() > 0.75:
# #         response = get_answer(tag)
# #         if response:
# #             print(f"{bot_name}: {response}")
# #         else:
# #             for intent in intents['intents']:
# #                 if tag == intent["tag"]:
# #                     print(f"{bot_name}: {random.choice(intent['responses'])}")
# #     else:
# #         print(f"{bot_name}: I do not understand...")
# #
# # conn.close()


import random
import json
import torch
import sqlite3

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Sam"
print("Let's chat! (type 'quit' to exit)")

conn = sqlite3.connect('menu.db')
cursor = conn.cursor()

def get_answer(tag):
    cursor.execute("SELECT price FROM drinks WHERE name=?", (tag,))
    result = cursor.fetchone()
    if result:
        return result[0]
    else:
        return None

while True:
    sentence = input("You: ")
    if sentence == "quit":
        break

    # Check if the user has asked for the price of an item
    if "price of" in sentence:
        # Get the item name from the sentence
        item = sentence.split("price of ")[1]
        # Query the database for the price of the item
        cursor.execute("SELECT price FROM drinks WHERE name=?", (item,))
        result = cursor.fetchone()
        if result is None:
            # If the item is not found in the database, respond accordingly
            print(f"{bot_name}: I'm sorry, I don't know the price of {item}.")
        else:
            # If the item is found in the database, give the price
            price = result[0]
            print(f"{bot_name}: The price of {item} is {price}.")

    else:
        sentence = tokenize(sentence)
        X = bag_of_words(sentence, all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(device)

        output = model(X)
        _, predicted = torch.max(output, dim=1)

        tag = tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]

        if prob.item() > 0.75:
            response = get_answer(tag)
            if response:
                print(f"{bot_name}: {response}")
            else:
                for intent in intents['intents']:
                    if tag == intent["tag"]:
                        print(f"{bot_name}: {random.choice(intent['responses'])}")
        else:
            print(f"{bot_name}: I do not understand...")
    #conn.close()




#     sentence = tokenize(sentence)
#     X = bag_of_words(sentence, all_words)
#     X = X.reshape(1, X.shape[0])
#     X = torch.from_numpy(X).to(device)
#
#     output = model(X)
#     _, predicted = torch.max(output, dim=1)
#
#     tag = tags[predicted.item()]
#
#     probs = torch.softmax(output, dim=1)
#     prob = probs[0][predicted.item()]
#
#     if prob.item() > 0.75:
#         response = get_answer(tag)
#         if response:
#             print(f"{bot_name}: {response}")
#         else:
#             for intent in intents['intents']:
#                 if tag == intent["tag"]:
#                     print(f"{bot_name}: {random.choice(intent['responses'])}")
#     else:
#         print(f"{bot_name}: I do not understand...")
#
# conn.close()