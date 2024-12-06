
import sys
from python_files.model import get_llm_models, create_basemodel, model_play_tic_tac_toe, get_ttt_position, ask_model
from langchain.schema.language_model import BaseLanguageModel
from pydantic import BaseModel, Field


class TTTPosition(BaseModel):
    """
    Return the position of the move in the Tic Tac Toe game as "X,Y" coordinates.
    """

    X: int = Field(..., description="The row number of the move.")
    Y: int = Field(..., description="The column number of the move.")


def get_args():
  all_llm_models = get_llm_models()
  llm_model = all_llm_models[0]
  board_max = 3
  game_type = "mvm"

  all_args = sys.argv[1:]

  if len(all_args) > 0:
    for arg in all_args:
      if arg.startswith("model="):
        llm_model = arg.split("=")[1]
      if arg.startswith("board_max="):
        board_max = int(arg.split("=")[1])
      if arg.startswith("mode="):
        game_type = arg.split("=")[1]
  return llm_model, board_max, game_type

if len(sys.argv) < 2:
  print("Using default game type 'mvm'.")
else:
  game_type = sys.argv[1]

def create_tic_tac_toe_board(board_max: int):
  # Create a 3x3 tic-tac-toe board
  board = []
  for i in range(board_max):
    board.append(['-'] * board_max)

  return board

def is_game_over(board: list, board_max: int) -> bool:
  """
  Vérifie s'il y a un gagnant dans une partie de morpion en comptant uniquement 3 cases.

  Args:
    board (list[list[str]]): Plateau de jeu.
    board_max (int): Taille du plateau (par exemple, 3 pour un morpion classique).

  Returns:
    bool: True s'il y a un gagnant, False sinon.
  """
  def check_three_in_a_row(line):
    for i in range(len(line) - 2):
      if line[i] == line[i + 1] == line[i + 2] and line[i] != '-':
        return True
    return False

  # Vérifie les lignes
  for row in board:
    if check_three_in_a_row(row):
      return True

  # Vérifie les colonnes
  for col in range(board_max):
    column = [board[row][col] for row in range(board_max)]
    if check_three_in_a_row(column):
      return True

  # Vérifie les diagonales
  for i in range(board_max - 2):
    main_diagonal = [board[i + j][j] for j in range(board_max - i)]
    anti_diagonal = [board[i + j][board_max - j - 1] for j in range(board_max - i)]
    if check_three_in_a_row(main_diagonal) or check_three_in_a_row(anti_diagonal):
      return True

  return False


def process_turn(board: list, player_form: str, position: str):
  row = int(position.split(",")[0]) - 1
  col = int(position.split(",")[1]) - 1
  board[row][col] = player_form
  return board

def print_board(board: list):
  header = "    " + "    ".join(str(i) for i in range(1, len(board) + 1))
  separator = "--+" + "----+" * len(board)
  rows = []
  for i, row in enumerate(board, start=1):
    rows.append(f"{i} | " + " | ".join(f" {cell}" for cell in row) + " |")
  print("\n".join([header, separator] + [f"{row}\n{separator}" for row in rows] + ["-" * (len(board)-1 * 5 + 3)]))

def is_position_valid(position: str, board_max: int):
  if len(position.split(",")) != 2:
    return False
  if not position.replace(",", "").isdigit():
    return False
  row = int(position.split(",")[0])
  col = int(position.split(",")[1])
  if row < 1 or row > board_max or col < 1 or col > board_max:
    return False
  return True

def is_valid_move(board: list, position: str):
  row = int(position.split(",")[0]) - 1
  col = int(position.split(",")[1]) - 1
  return board[row][col] == '-'

def get_invalid_position(board: list, board_max: int):
  invalid_pos = ""
  for i in range(board_max):
    for j in range(board_max):
      if board[i][j] != "-":
        invalid_pos += f"{i+1},{j+1}" + "\n"
  return invalid_pos

def player_turn(boardgame: list, player_form: str, board_max: int):
  position = ""
  while True:
    position = input("Player 1, enter your position (X,Y): ")
    if is_position_valid(position, board_max) and is_valid_move(boardgame, position):
      boardgame = process_turn(boardgame, player_form, position)
      return boardgame

def model_turn(llm_model: BaseLanguageModel, boardgame: list, player_form: str, board_max: int):
  invalid_position = get_invalid_position(boardgame, board_max)
  while True:
    position = model_play_tic_tac_toe(llm_model, boardgame, player_form, invalid_position, board_max)
    # print("Player 2 position:", position)
    if is_position_valid(position, board_max) and is_valid_move(boardgame, position):
      boardgame = process_turn(boardgame, player_form, position)
      return boardgame

async def start_game(llm_model_names: str = "llama3.2:latest", board_max: int = 3, game_type: str = "mvm"):
  player_first = False
  player_vs_player = False
  player_vs_model = False
  model_vs_model = False

  llm_model_names, board_max, game_type = get_args()
  print("Playing with LLM models:", llm_model_names)
  print("Board size:", board_max)
  print("Game type:", game_type)

  tools = [TTTPosition, get_ttt_position]

  # check if there is a "," in the model name
  if "," in llm_model_names:
    llm_model1_name, llm_model2_name = llm_model_names.split(",")
  else:
    llm_model1_name = llm_model_names
    llm_model2_name = llm_model_names


  llm_model1, llm_model2 = None, None

  if game_type == "pvp":
    player_vs_player = True
  elif game_type == "pvm":
    print("Playing against LLM model:", llm_model1_name)
    player_vs_model = True
    llm_model1 = create_basemodel(llm_model1_name, tools)
  elif game_type == "mvm":
    model_vs_model = True
    llm_model1 = create_basemodel(llm_model1_name, tools)
    llm_model2 = create_basemodel(llm_model2_name, tools)
  else:
    print("Invalid game type. Please enter 'pvp', 'pvm', 'mvm'.")
    return

  llm_model = llm_model1

  
  boardgame = create_tic_tac_toe_board(board_max)
  print_board(boardgame)

  player_form = 'X'
  turn_count = 0

  while True:
    turn_count += 1
    if player_vs_player:
      boardgame = player_turn(boardgame, player_form, board_max)
    elif model_vs_model:
      print("Model playing: ", llm_model)
      boardgame = model_turn(llm_model, boardgame, player_form, board_max)
      llm_model = llm_model2 if llm_model == llm_model1 else llm_model1
    elif player_vs_model:
      if player_first:
        boardgame = player_turn(boardgame, player_form, board_max)
      else:
        boardgame = model_turn(llm_model, boardgame, player_form, board_max)
      player_first = not player_first

    print(f"Turn {turn_count}")
    print_board(boardgame)

    if is_game_over(boardgame, board_max):
      print(f"Player {player_form} wins!")
      break
    
    player_form = 'O' if player_form == 'X' else 'X'


def main():
  start_game()

if __name__ == "__main__":
  main()