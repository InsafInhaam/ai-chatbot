import sqlite3

# create a connection to the database
conn = sqlite3.connect('menu.db')

# create a table for drinks
conn.execute('''CREATE TABLE drinks
             (name TEXT PRIMARY KEY,
              price REAL);''')

# insert some data into the table
conn.execute("INSERT INTO drinks (name, price) VALUES ('coffee', 2.5)")
conn.execute("INSERT INTO drinks (name, price) VALUES ('tea', 1.5)")
conn.execute("INSERT INTO drinks (name, price) VALUES ('latte', 3.0)")

# commit changes and close the connection
conn.commit()
conn.close()
