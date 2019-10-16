# solutions.py
"""Volume 3: Web Technologies. Solutions File.
<Mark Rose>
<Section 2>
<9/16/19>
"""


import json
import socket
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


# Problem 1
def prob1(filename="nyc_traffic.json"):
    """Load the data from the specified JSON file. Look at the first few
    entries of the dataset and decide how to gather information about the
    cause(s) of each accident. Make a readable, sorted bar chart showing the
    total number of times that each of the 7 most common reasons for accidents
    are listed in the data set.
    """
    #load the file and initialize a list of crashes
    crash_reasons = []
    with open(filename) as my_file:
        crash_data = json.load(my_file)
        
    #Get the crash reasons
    for i in crash_data:
        if 'contributing_factor_vehicle_1' in i:
            crash_reasons.append(i['contributing_factor_vehicle_1'])
        if 'contributing_factor_vehicle_2' in i:
            crash_reasons.append(i['contributing_factor_vehicle_2'])
        if 'contributing_factor_vehicle_3' in i:
            crash_reasons.append(i['contributing_factor_vehicle_3'])
        if 'contributing_factor_vehicle_4' in i:
            crash_reasons.append(i['contributing_factor_vehicle_4'])
        if 'contributing_factor_vehicle_5' in i:
            crash_reasons.append(i['contributing_factor_vehicle_5'])
    
    #get the counts of each of the reasons and sort
    counts = Counter(crash_reasons)
    labels, values = zip(*counts.items())
    indSort = np.argsort(values)[::-1]
    
    #Get the labels and values
    labels = np.array(labels)[indSort][:7]
    values = np.array(values)[indSort][:7]
    indexes = np.arange(7)

    #Set all of the plot attributes and show it
    bar_width = .35
    plt.bar(indexes, values)
    plt.xticks(indexes, labels, rotation='vertical')
    plt.xlabel("Crash Reasons")
    plt.ylabel("Counts")
    plt.title("Amount of Crash Reasons in New York")
    plt.tight_layout()
    plt.show()
    return


class TicTacToe:
    def __init__(self):
        """Initialize an empty board. The O's go first."""
        self.board = [[' ']*3 for _ in range(3)]
        self.turn, self.winner = "O", None

    def move(self, i, j):
        """Mark an O or X in the (i,j)th box and check for a winner."""
        if self.winner is not None:
            raise ValueError("the game is over!")
        elif self.board[i][j] != ' ':
            raise ValueError("space ({},{}) already taken".format(i,j))
        self.board[i][j] = self.turn

        # Determine if the game is over.
        b = self.board
        if any(sum(s == self.turn for s in r)==3 for r in b):
            self.winner = self.turn     # 3 in a row.
        elif any(sum(r[i] == self.turn for r in b)==3 for i in range(3)):
            self.winner = self.turn     # 3 in a column.
        elif b[0][0] == b[1][1] == b[2][2] == self.turn:
            self.winner = self.turn     # 3 in a diagonal.
        elif b[0][2] == b[1][1] == b[2][0] == self.turn:
            self.winner = self.turn     # 3 in a diagonal.
        else:
            self.turn = "O" if self.turn == "X" else "X"

    def empty_spaces(self):
        """Return the list of coordinates for the empty boxes."""
        return [(i,j) for i in range(3) for j in range(3)
                                        if self.board[i][j] == ' ' ]
    def __str__(self):
        return "\n---------\n".join(" | ".join(r) for r in self.board)


# Problem 2
class TicTacToeEncoder(json.JSONEncoder):
    """A custom JSON Encoder for TicTacToe objects."""
    def default(self, obj):
        #If not a tictactoe object raise an error
        if not isinstance(obj, TicTacToe):
            raise TypeError("expected a TicTacToe object for encoding")
        #Create a json encoder
        return {"dtype": "TicTacToe", "board": obj.board, "turn": obj.turn, "winner": obj.winner}


# Problem 2
def tic_tac_toe_decoder(obj):
    """A custom JSON decoder for TicTacToe objects."""
    #Decode the tic_tac_toe message and update its attributes
    if "dtype" in obj:
        new_tic = TicTacToe()
        new_tic.board = obj['board']
        new_tic.turn = obj['turn']
        new_tic.winner = obj['winner']
        return new_tic
    raise ValueError("expected a TicTacToe object")
    pass


def mirror_server(server_address=("0.0.0.0", 33333)):
    """A server for reflecting strings back to clients in reverse order."""
    print("Starting mirror server on {}".format(server_address))

    # Specify the socket type, which determines how clients will connect.
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.bind(server_address)    # Assign this socket to an address.
    server_sock.listen(1)               # Start listening for clients.

    while True:
        # Wait for a client to connect to the server.
        print("\nWaiting for a connection...")
        connection, client_address = server_sock.accept()

        try:
            # Receive data from the client.
            print("Connection accepted from {}.".format(client_address))
            in_data = connection.recv(1024).decode()    # Receive data.
            print("Received '{}' from client".format(in_data))

            # Process the received data and send something back to the client.
            out_data = in_data[::-1]
            print("Sending '{}' back to the client".format(out_data))
            connection.sendall(out_data.encode())       # Send data.

        # Make sure the connection is closed securely.
        finally:
            connection.close()
            print("Closing connection from {}".format(client_address))

def mirror_client(server_address=("0.0.0.0", 33333)):
    """A client program for mirror_server()."""
    print("Attempting to connect to server at {}...".format(server_address))

    # Set up the socket to be the same type as the server.
    client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_sock.connect(server_address)    # Attempt to connect to the server.
    print("Connected!")
    
    # Send some data from the client user to the server.
    out_data = input("Type a message to send to the server: ")
    client_sock.sendall(out_data.encode())              # Send data.

    # Wait to receive a response back from the server.
    in_data = client_sock.recv(1024).decode()           # Receive data.
    print("Received '{}' from the server".format(in_data))

    # Close the client socket.
    client_sock.close()


# Problem 3
def tic_tac_toe_server(server_address=("0.0.0.0", 44444)):
    """A server for playing tic-tac-toe with random moves."""
    print("Starting TicTacToe server on {}".format(server_address))

    # Specify the socket type, which determines how clients will connect.
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.bind(server_address)    # Assign this socket to an address.
    server_sock.listen(1)               # Start listening for clients.
    
    # Wait for a client to connect to the server.
    print("\nWaiting for a connection...")
    connection, client_address = server_sock.accept()

    # Receive data from the client.
    print("Connection accepted from {}.".format(client_address))
    
    while True:
        #Recieve Data
        in_data = connection.recv(1024).decode()          
        my_tic = json.loads(in_data, object_hook=tic_tac_toe_decoder)
        
        # Process the received data and send something back to the client.
        move_list = my_tic.empty_spaces()
        
        #If there is a winner, send a message and the board
        if my_tic.winner == 'O':
            connection.sendall('You WIN'.encode())
            out_data = json.dumps(my_tic, cls = TicTacToeEncoder)
            connection.sendall(out_data.encode())
            break
        #If there is no more room in the board, but no winner (i.e. a tie) then send a message and the board
        elif len(move_list) == 0:
            connection.sendall('We DRAW'.encode())
            out_data = json.dumps(my_tic, cls = TicTacToeEncoder)
            connection.sendall(out_data.encode())
            break
        #Otherwise make a move
        else:
            move_list = my_tic.empty_spaces()
            i,j = move_list[10%len(move_list)]
            my_tic.move(i,j)
            #If this move caused the server to win, tell the client they lost and send them the board
            if my_tic.winner == 'X':
                connection.sendall('You LOSE'.encode())
                out_data = json.dumps(my_tic, cls = TicTacToeEncoder)
                connection.sendall(out_data.encode()) 
                break
            #Otherwise send jsut the board
            else:
                out_data = json.dumps(my_tic, cls = TicTacToeEncoder)
                connection.sendall(out_data.encode())  
    #Close the connecton once the game is over
    connection.close()
    print("Closing connection from {}".format(client_address))
    return


# Problem 4
def tic_tac_toe_client(server_address=("0.0.0.0", 44444)):
    """A client program for tic_tac_toe_server()."""
    
    #Connect to the server at the server_address provided
    print("Attempting to connect to server at {}...".format(server_address))

    # Set up the socket to be the same type as the server.
    client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_sock.connect(server_address)    # Attempt to connect to the server.
    print("Connected!")
    
    #Initialize the tictactoe game
    my_tic = TicTacToe()
    
    while True:
        #Print the tictactoe board nicely for the client to see
        print(my_tic.board[0])
        print(my_tic.board[1])
        print(my_tic.board[2])
        
        #Prompt the user for their move and reprompt if they give an invalid move
        while True:
            my_move = input("Your Move: ")
            i = int(my_move[0])
            j = int(my_move[-1])
            if (i,j) not in my_tic.empty_spaces():
                print('Bad Input. Try Again.')
            else:
                break
        #Make your move and send the data to the server
        my_tic.move(i,j)
        out_data = json.dumps(my_tic, cls = TicTacToeEncoder)
        client_sock.sendall(out_data.encode())              # Send data.

        #Recieve data from the server
        in_data = client_sock.recv(1024).decode()           
        
        #If any of the data is a message other than the board recieve that message and then recieve the board
        if in_data == "You WIN" or in_data == 'We DRAW' or in_data == 'You LOSE':
            print('\n' + in_data)
            in_data = client_sock.recv(1024).decode()
            my_tic = json.loads(in_data, object_hook=tic_tac_toe_decoder)
            
            #Print the end board and close the connectioin
            print(my_tic.board[0])
            print(my_tic.board[1])
            print(my_tic.board[2])
            client_sock.close()
            break
        else:
            #Otherwise get the tictactoe board
            my_tic = json.loads(in_data, object_hook=tic_tac_toe_decoder)
    return


#server_address=("0.0.0.0", 30000)

