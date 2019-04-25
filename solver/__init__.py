import random

def getNextNode(Questions, Responses, GraphicalSimilarity , SemanticSimilarity) -> int:

    max_val = len(GraphicalSimilarity)-1
    selected = random.randint(0, max_val)
    return selected



