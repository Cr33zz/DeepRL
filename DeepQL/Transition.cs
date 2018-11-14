using Neuro.Tensors;

namespace DeepQL
{
    public class Transition
    {
        public Transition(Tensor state, Tensor action, double reward, Tensor nextState, bool done)
        {
            State = state;
            Action = action;
            NextState = nextState;
            Reward = reward;
            Done = done;
        }

        public readonly Tensor State;
        public readonly Tensor Action;
        public readonly Tensor NextState;
        public readonly double Reward;
        public readonly bool Done;
    }
}
