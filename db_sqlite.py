# #from sqlalchemy import create_engine
# import sqlite3
# #engine = create_engine('sqlite:///coffee_shop.db')
# #conn = engine.connect()
# # Connect to the database
# conn = sqlite3.connect('coffee_shop.db')
# c = conn.cursor()
#
#
# def get_response(query):
#     # Execute the query and fetch all results
#     c.execute(query)
#     results = c.fetchall()
#     # If there are no results, return None
#     if len(results) == 0:
#         return None
#     # Otherwise, format the results as a string and return
#     else:
#         return ', '.join([str(r[0]) for r in results])
#
import sqlite3

# create a connection to the database
conn = sqlite3.connect('menu.db')

# define a function to retrieve the list of drinks
def get_drinks():
    # execute a query to retrieve the list of drinks
    cursor = conn.execute("SELECT name FROM drinks")
    # extract the data from the query results
    drinks = [row[0] for row in cursor]
    # return the list of drinks
    return drinks

# define a function to retrieve the price of a drink
def get_price(drink):
    print(f"get_price called with drink={drink}")
    # execute a query to retrieve the price of the specified drink
    cursor = conn.execute('''SELECT price FROM drinks WHERE name = ?''', (drink,))
    # extract the data from the query results
    price = cursor.fetchone()
    # return the price of the drink (or None if not found)
    return price[0] if price else None

# define a function to handle user input and generate a response
def respond(message):
    # normalize the user input
    message = message.lower().strip()
    # handle different types of messages
    if message == 'hello':
        return "Hi there, how can I help you?"
    elif message == 'what drinks do you have?':
        drinks = get_drinks()
        return "We have the following drinks: " + ', '.join(drinks)
    elif message.startswith('how much is'):
        drink = message[10:].strip()
        price = get_price(drink)
        if price:
            return f"The price of {drink} is {price} dollars."
        else:
            return f"Sorry, we don't have {drink} in our menu."
    else:
        return "Sorry, I don't understand. Please try again."
 # Commit your changes in the database
conn.commit()

#Closing the connection
conn.close()
# run a loop to handle user input and generate responses
# while True:
#     message = input("You: ")
#     response = respond(message)
#     print("Bot:", response)
