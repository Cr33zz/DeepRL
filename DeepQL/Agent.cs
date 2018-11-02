using Neuro.Tensors;
using DeepQL.Models;

namespace DeepQL
{
    public abstract class Agent
    {
        public Agent(ModelBase model, Environment env)
        {
            Model = model;
            Env = env;
        }

        public int ChooseNextAction(Tensor state)
        {
            return 0;
        }

        public void TakeAction(int action)
        {
            var nextState = new Tensor(Env.StateShape());
            var reward = Env.Step(action, nextState);

            Model.OnTransition(LastState, action, reward, nextState);

            LastState = nextState;
        }

        private Tensor LastState;
        private ModelBase Model;
        private Environment Env;
    }
}
