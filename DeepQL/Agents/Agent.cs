using System;
using System.Threading;
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

        public abstract void Train(int episodes, int maxStepsPerEpisode, bool render);

        public abstract double Test(int episodes, int maxStepsPerEpisode, bool render);

        //public void TakeAction()
        //{
        //    int action = ChooseNextAction();
        //    var stop = Env.Step(action, out var nextState, out var reward);

        //    OnStep(State, action, reward, nextState);

        //    State = nextState;
        //}

        protected abstract int ChooseNextAction();

        protected abstract void OnStep(Tensor state, int action, double reward, Tensor nextState);

        protected void Render(bool render)
        {
            if (render)
            {
                Env.Render();
                Thread.Sleep(1000 / StepsPerSec);
            }
        }

        protected Tensor LastObservation;
        protected readonly Env Env;
        
        protected double Epsilon; // Exploration probability
        protected double MinEpsilon = 0.01;
        protected double MaxEpsilon = 1.0;
        protected double EpsilonDecay = 0.995;

        public bool Verbose = false;
        public int StepsPerSec = 30;
    }
}
