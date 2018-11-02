using Neuro.Tensors;
using DeepQL.Q;

namespace DeepQL
{
    public abstract class Agent
    {
        public Agent(QFunc qFunction, Environment env)
        {
            QFunction = qFunction;
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

            QFunction.OnTransition(LastState, action, reward, nextState);

            LastState = nextState;
        }

        private Tensor LastState;
        private QFunc QFunction;
        private Environment Env;
    }
}
