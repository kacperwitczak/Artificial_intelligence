import copy


class AlphaBetaAgent:
    def __init__(self, my_token, heuristic = True,depth=5):
        self.my_token = my_token
        self.enemy_token = 'o' if self.my_token != 'o' else 'x'
        self.depth = depth
        self.heuristic = heuristic

    def decide(self, connect4):
        best_value = -float("inf")
        best_move = None

        #find best move
        for drop in connect4.possible_drops():
            connect4_copy = copy.deepcopy(connect4)
            connect4_copy.drop_token(drop)
            value = self.minimax(connect4_copy, False, self.depth, -float('inf'), float('inf'))

            if value > best_value:
                best_value = value
                best_move = drop

        return best_move


    def calculate_tokens(self, token):
        agent = self.my_token
        enemy = self.enemy_token
        empty = '_'
        agent_tokens = 0
        enemy_tokens = 0
        empty_tokens = 0
        for t in token:
            if t == agent:
                agent_tokens+=1
            elif t==enemy:
                enemy_tokens+=1
            elif t==empty:
                empty_tokens+=1

        return agent_tokens, enemy_tokens, empty_tokens

    def evaluate_state(self, state):
        score = 0
        counter = 0
        for four in state.iter_fours():
            counter += 1

            agent_count, enemy_count, empty_count = self.calculate_tokens(four)

            if agent_count == 3 and empty_count == 1:
                score += 2
            elif enemy_count == 3 and empty_count == 1:
                score -= 2
            elif agent_count == 2 and empty_count == 2:
                score += 0.3
            elif enemy_count == 2 and empty_count == 2:
                score -= 0.3
            elif agent_count == 1 and empty_count == 3:
                score += 0.1
            elif enemy_count == 1 and empty_count == 3:
                score -= 0.1

        state_value = (score / counter)

        return state_value

    def minimax(self, state, maximizingPlayer, depth, alpha, beta):
        if state.game_over:
            if state.wins == self.my_token:
                return 1*depth
            elif state.wins is None:
                return 0
            else:
                return -1*depth
        elif self.heuristic and depth == 0:
            return self.evaluate_state(state)
        elif not self.heuristic and depth == 0:
            return 0

        elif maximizingPlayer:
            value = -float('inf')
            for drop in state.possible_drops():
                state_copy = copy.deepcopy(state)
                state_copy.drop_token(drop)
                value = max(value, self.minimax(state_copy, False, depth-1, alpha, beta))
                alpha = max(alpha, value)
                if value >= beta:
                    break

            return value

        else:
            value = float('inf')
            for drop in state.possible_drops():
                state_copy = copy.deepcopy(state)
                state_copy.drop_token(drop)
                value = min(value, self.minimax(state_copy, True, depth-1, alpha, beta))
                beta = min(beta, value)
                if value <= alpha:
                    break

            return value

