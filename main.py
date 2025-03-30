from model import Model
import json

def main():
    with open("config.json", "r") as file:
        config = json.load(file)
    with open("chat_history", "r") as file:
        previous_dialog = file.read()
    model = Model(huggingface_key=config["huggingface_key"], previous_dialog=previous_dialog)

    while True:
        input_text = input("> ")
        if input_text == "stop":
            break
        response = model.interact(input_text)
        print("<", response)

    model.train_on_interaction()
    
if __name__=="__main__":
    main()