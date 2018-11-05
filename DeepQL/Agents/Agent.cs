using System;
using DeepQL.Environments;
using Neuro.Tensors;
using DeepQL.Q;

namespace DeepQL.Agents
{
    public abstract class Agent
    {
        protected Agent(Env env, double learningRate, double discoutFactor, bool verbose = false)
        {
            Env = env;
            LearningRate = learningRate;
            DiscountFactor = discoutFactor;
            Epsilon = MaxEpsilon;
            Verbose = verbose;
        }

        public abstract void Train(int episodes, int maxStepsPerEpisode);

        public abstract double Test(int episodes, int maxStepsPerEpisode);

        public void TakeAction()
        {
            int action = ChooseNextAction();
            var stop = Env.Step(action, out var nextState, out var reward);

            OnStep(State, action, reward, nextState);

            State = nextState;
        }

        protected abstract int ChooseNextAction();

        protected abstract void OnStep(Tensor state, int action, double reward, Tensor nextState);

        protected Tensor State;
        protected readonly Env Env;
        protected readonly double LearningRate;
        protected readonly double DiscountFactor;

        protected double Epsilon; // Exploration probability
        protected double MinEpsilon = 0.01;
        protected double MaxEpsilon = 1.0;
        protected double EpsilonDecay = 0.01;

        protected readonly bool Verbose;
    }
}
