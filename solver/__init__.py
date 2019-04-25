import random

def getNextNode(Questions, Responses, GraphicalSimilarity , SemanticSimilarity) -> int:
    return random.randint(0, len(GraphicalSimilarity)-1)


