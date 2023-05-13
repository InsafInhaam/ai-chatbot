# # import sqlite3
# # import torch
# # import torch.nn as nn
# #
# #
# #
# # # Define your PyTorch model
# # class Chatbot(nn.Module):
# #     def __init__(self, input_size, hidden_size, output_size):
# #         super(MyModel, self).__init__()
# #         self.fc1 = nn.Linear(input_size, hidden_size)
# #         self.relu = nn.ReLU()
# #         self.fc2 = nn.Linear(hidden_size, output_size)
# #     def forward(self, x):
# #         out = self.fc1(x)
# #         out = self.relu(out)
# #         out = self.fc2(out)
# #         return out
# #
# #
# # # Connect to your SQLite database
# # conn = sqlite3.connect('menu.db')
# # cur = conn.cursor()
# #
# #
# #
# # # Retrieve relevant data from the database based on user input
# # user_input = input('User: ')
# # query = f"SELECT * FROM drinks WHERE price = '{user_input}'"
# # cur.execute(query)
# # result = cur.fetchone()
# #
# #
# #
# # # Load the trained model
# # model = torch.load('chatbot.pt')
# #
# #
# #
# # # Use the model to generate a response
# # if result is not None:
# #     input_data = result[1]
# #     output_data = result[2]
# #     input_tensor = torch.tensor(input_data)
# #     output_tensor = model(input_tensor)
# #     response = output_tensor.tolist()
# #     print(f"ChatBot: {response}")
# #
# # else:
# #     print("ChatBot: I'm sorry, I don't understand.")
# import sqlite3
# import sqlite3
# import torch
# import torch.nn as nn
# import numpy as np
#
# # Define your PyTorch model
# class ChatBot(nn.Module):
#     def _init_(self, input_size, hidden_size, output_size):
#         super(ChatBot, self)._init_()
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.output_size = output_size
#         self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
#         self.fc = nn.Linear(hidden_size, output_size)
#
#     def forward(self, input):
#         hidden = torch.zeros(1, input.size(0), self.hidden_size).to(input.device)
#         output, hidden = self.gru(input, hidden)
#         output = self.fc(output[:, -1, :])
#         return output
#
# # Hyperparameters
# input_size = 10
# hidden_size = 20
# output_size = 5
# learning_rate = 0.01
# num_epochs = 100
#
# # Connect to your SQLite database
# conn = sqlite3.connect('menu.db')
# cur = conn.cursor()
#
# # Retrieve relevant data from the database based on user input
# def get_data(user_input):
#     query = f"SELECT * FROM drinks WHERE price = '{user_input}'"
#     cur.execute(query)
#     result = cur.fetchone()
#     return result
#
# # Load the trained model
# model = ChatBot(input_size, hidden_size, output_size)
# model.load_state_dict(torch.load('chatbot.pt'))
#
# # Generate a response to user input
# def generate_response(user_input):
#     result = get_data(user_input)
#     if result is not None:
#         input_data = np.frombuffer(result[1], dtype=np.float32).reshape(1, -1)
#         input_tensor = torch.from_numpy(input_data)
#         output_tensor = model(input_tensor)
#         response = np.array2string(output_tensor.detach().numpy(), separator=',')
#         response = response.replace('[', '').replace(']', '').replace(',', '')
#         print(f"ChatBot: {response}")
#     else:
#         print("ChatBot: I'm sorry, I don't understand.")
#
# # Main loop
# while True:
#     user_input = input('User: ')
#     generate_response(user_input)