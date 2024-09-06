from AI.persistent_memory import VectorDB


class Memory:
    def __init__(self):
        self.vectordb = VectorDB()