using System;
using DeepQL.Environments;
using Neuro.Tensors;

namespace DeepQL.Agents
{
    public abstract class Agent
    {
        protected Agent(Env env, bool verbose = false)
        {
            Env = env;
            Epsilon = MaxEpsilon;
            Verbose = verbose;
        }

        public abstract void Train(int episodes, int maxStepsPerEpisode);

        public abstract double Test(int episodes, int maxStepsPerEpisode);

        //public void TakeAction()
        //{
        //    int action = ChooseNextAction();
        //    var stop = Env.Step(action, out var nextState, out var reward);

        //    OnStep(State, action, reward, nextState);

        //    State = nextState;
        //}

        protected abstract int ChooseNextAction();

        protected abstract void OnStep(Tensor state, int action, double reward, Tensor nextState);

        protected Tensor LastObservation;
        protected readonly Env Env;
        
        protected double Epsilon; // Exploration probability
        protected double MinEpsilon = 0.01;
        protected double MaxEpsilon = 1.0;
        protected double EpsilonDecay = 0.01;

        protected readonly bool Verbose;
    }
}
