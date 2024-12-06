## Run web serveur
make run

URL: http://127.0.0.1:8000/ask
=> allows user to ask a question to the LLM

URL: http://127.0.0.1:8000/search
=> allows user to ask a question to the LLM using an URL containing JSON data to retrieve data from it


## Tic-Tac-Toe game

```bash
# Play tic-tac-toe player vs player
python python_files/game.py pvp

# Play tic-tac-toe player vs model
python python_files/game.py pvm

# Play tic-tac-toe model vs model
python python_files/game.py mvm

```
