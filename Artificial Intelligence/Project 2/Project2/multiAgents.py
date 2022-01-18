# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        newFood = newFood.asList()
        ghostPos = [(G.getPosition()[0], G.getPosition()[1]) for G in newGhostStates]
        scared = newScaredTimes[0] > 0

        if not scared and (newPos in ghostPos):
            return -1.0
        if newPos in currentGameState.getFood().asList():
            return 1

        closestFoodDist = sorted(newFood, key=lambda fD: util.manhattanDistance(fD, newPos))
        closestGhostDist = sorted(ghostPos, key=lambda gD: util.manhattanDistance(gD, newPos))

        fd = lambda fDis: util.manhattanDistance(fDis, newPos)
        gd = lambda gDis: util.manhattanDistance(gDis, newPos)

        return 1/fd(closestFoodDist[0]) - 1/gd(closestGhostDist[0])

        # return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        GhostIdx = [i for i in range(1, gameState.getNumAgents())]

        def minimum(state, d, ghost):
            if state.isWin() or state.isLose() or d == self.depth:
                return self.evaluationFunction(state)

            v = float('inf')
            for action in state.getLegalActions(ghost):
                if ghost == GhostIdx[-1]:
                    v = min(v, maximum(state.generateSuccessor(ghost, action), d+1))
                else:
                    v = min(v, minimum(state.generateSuccessor(ghost, action), d, ghost+1))
            return v

        def maximum(state, d):
            if state.isWin() or state.isLose() or d == self.depth:
                return self.evaluationFunction(state)

            v = float('-inf')
            for action in state.getLegalActions(0):
                v = max(v, minimum(state.generateSuccessor(0, action), d, 1))
            return v

        ret = [(action, minimum(gameState.generateSuccessor(0, action), 0, 1)) for action in gameState.getLegalActions(0)]
        ret.sort(key=lambda k: k[1])

        return ret[-1][0]

        # util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        GhostIdx = [i for i in range(1, gameState.getNumAgents())]

        def minimum(state, d, ghost, A, B):
            if state.isWin() or state.isLose() or d == self.depth:
                return self.evaluationFunction(state)

            v = float('inf')
            for action in state.getLegalActions(ghost):
                if ghost == GhostIdx[-1]:
                    v = min(v, maximum(state.generateSuccessor(ghost, action), d+1, A, B))
                else:
                    v = min(v, minimum(state.generateSuccessor(ghost, action), d, ghost+1, A, B))

                if v < A:
                    return v

                B = min(B, v)
            return v

        def maximum(state, d, A, B):
            if state.isWin() or state.isLose() or d == self.depth:
                return self.evaluationFunction(state)

            v = float('-inf')
            for action in state.getLegalActions(0):
                v = max(v, minimum(state.generateSuccessor(0, action), d, 1, A, B))

                if v > B:
                    return v

                A = max(A, v)
            return v

        def alphabeta(state):
            v = float('-inf')
            action = None
            A = float('-inf')
            B = float('inf')

            for a in state.getLegalActions(0):
                temp = minimum(gameState.generateSuccessor(0, a), 0, 1, A, B)

                if v < temp:
                    v = temp
                    action = a

                if v > B:
                    return v

                A = max(A, temp)
            return action

        return alphabeta(gameState)

        # util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        GhostIdx = [i for i in range(1, gameState.getNumAgents())]

        def exp(state, d, ghost):
            if state.isWin() or state.isLose() or d == self.depth:
                return self.evaluationFunction(state)

            v = 0
            prob = 1/len(state.getLegalActions(ghost))

            for action in state.getLegalActions(ghost):
                if ghost == GhostIdx[-1]:
                    v += prob * maximum(state.generateSuccessor(ghost, action), d+1)
                else:
                    v += prob * exp(state.generateSuccessor(ghost, action), d, ghost+1)
            return v

        def maximum(state, d):
            if state.isWin() or state.isLose() or d == self.depth:
                return self.evaluationFunction(state)

            v = float('-inf')
            for action in state.getLegalActions(0):
                v = max(v, exp(state.generateSuccessor(0, action), d, 1))
            return v

        ret = [(action, exp(gameState.generateSuccessor(0, action), 0, 1)) for action in gameState.getLegalActions(0)]
        ret.sort(key=lambda k: k[1])
        return ret[-1][0]

        # util.raiseNotDefined()

def betterEvaluationFunction(currentGameState, ghosts=None):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: try to make pacman more scared to the ghost, and make more continuous eating rather than long "no point" move

    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()

    newFood = newFood.asList()
    ghostPos = [(G.getPosition()[0], G.getPosition()[1]) for G in newGhostStates]

    if currentGameState.isLose():
        return float('-inf')
    if newPos in ghostPos:
        return float('-inf')

    closestFoodDist = sorted(newFood, key=lambda fD: util.manhattanDistance(fD, newPos))
    closestGhostDist = sorted(ghostPos, key=lambda gD: util.manhattanDistance(gD, newPos))

    fd = lambda fDis: util.manhattanDistance(fDis, newPos)
    gd = lambda gDis: util.manhattanDistance(gDis, newPos)

    # Having the pacman stay a little far away from the ghosts
    score = 0
    if gd(closestGhostDist[0]) < 3:
        score -= 100
    if gd(closestGhostDist[0]) < 2:
        score -= 200
    if gd(closestGhostDist[0]) < 1:
        score -= 500

    # Encourage the pacman to eat the capsules as much as possible without break
    if len(currentGameState.getCapsules()) == 1:
        score += 100

    if len(closestFoodDist) == 0 or len(closestGhostDist) == 0:
        score += scoreEvaluationFunction(currentGameState) + 1
    else:
        score += scoreEvaluationFunction(currentGameState) + 10/fd(closestFoodDist[0]) + 1/gd(closestGhostDist[0]) + 1/gd(closestGhostDist[-1])

    return score


# Abbreviation
better = betterEvaluationFunction
