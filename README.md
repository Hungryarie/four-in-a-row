# four-in-a-row
Training and playing Four In A Row by the means of a neural net.

## How to play:
- Open playgame.py
    - Chose which functions you want to run in the if __name__ == '__main__' part.
        - TestInEnv()
        - PlayInEnv()
        - trainNN()
        
- Select your players in every function: 
    - px = players.Human()
    - px = players.Drunk()
    - px = players.DDQNPlayer(Model). 
        - Where Model = modelX(input_shape=(6, 7, 1), output_num=7) 
        - OR where Model = load_a_model('models\x.model')

## what works:
- playing against a dump computer (plays random)
- training pipeline for DDQN reinforcement learning model


## work to be done:
- optimizing the reinforcement learning model
