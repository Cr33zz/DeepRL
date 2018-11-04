using DeepQL.Environments;
using Neuro.Tensors;
using DeepQL.Q;

namespace DeepQL
{
    public abstract class Agent
    {
        public Agent(QFunc qFunction, Env env)
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
            Tensor nextState;
            double reward;
            var stop = Env.Step(action, out nextState, out reward);

            QFunction.OnTransition(LastState, action, reward, nextState);

            LastState = nextState;
        }

        private Tensor LastState;
        private QFunc QFunction;
        private Env Env;
    }
}
