using Neuro.Tensors;

namespace DeepQL
{
    public class Transition
    {
        public Transition(Tensor state, int action, double reward, Tensor nextState)
        {
            State = state;
            Action = action;
            NextState = nextState;
            Reward = reward;
        }

        public readonly Tensor State;
        public readonly int Action;
        public readonly Tensor NextState;
        public readonly double Reward;
    }
}
