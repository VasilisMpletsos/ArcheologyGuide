from typing import Union
from fastapi import FastAPI

app = FastAPI()

@app.get("/building/{building_id}")
def get_building_general_informations(building_id: int):
    building1_story = '''This is the story of building 1, it was used as a sancuary for the people of the city ok kythnos. It is told that god Dimitra was worshiped there.'''
    building2_story = '''This is the story of building 2. It is the place where the people of the temple where living.'''
    if building_id == 1:
        return {"story": building1_story}
    elif building_id == 2:
        return {"story": building2_story}
    else:
        return {"story": None}

@app.get("/questions/{building_id}")
def read_item(building_id: int, question: Union[str, None] = None):
    return {"item_id": building_id, "question": question}