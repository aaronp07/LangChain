import sqlite3

# Connect to sqlite
connection = sqlite3.connect('student.db')

# Create a cursor to create table, insert record
cursor = connection.cursor()

# Create table
table_info = """
create table STUDENT(NAME VARCHAR(25), CLASS VARCHAR(25), SECTION VARCHAR(25), MARKS INT)
"""

cursor.execute(table_info)

# Insert some more records
cursor.execute('''insert into STUDENT values('Krish', 'Data Science', 'A', 90)''')
cursor.execute('''insert into STUDENT values('John', 'Data Science', 'B', 100)''')
cursor.execute('''insert into STUDENT values('Mukesh', 'Data Science', 'A', 86)''')
cursor.execute('''insert into STUDENT values('Jacob', 'DevOps', 'A', 50)''')
cursor.execute('''insert into STUDENT values('Dipesh', 'DevOps', 'A', 35)''')

# Display all the records
print('The inserted records are')
data = cursor.execute('''select * from STUDENT''')

for row in data:
    print(row)
    
# Commit the changes in the database
connection.commit()
connection.close()