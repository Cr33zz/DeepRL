using System;
using System.Linq;
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

        public void Train(int episodes, int maxStepsPerEpisode, bool render)
        {
            for (int ep = 0; ep < episodes; ++ep)
            {
                LastObservation = Env.Reset();

                double totalReward = 0;

                for (int step = 0; step < maxStepsPerEpisode; ++step)
                {
                    Tensor action;

                    if (GlobalRandom.Rng.NextDouble() < Epsilon)
                        action = Env.ActionSpace.Sample(); // explore
                    else
                        action = GetOptimalAction(); // exploit

                    bool done = Env.Step(action, out var observation, out var reward);
                    totalReward += reward;

                    Render(render);

                    OnStep(step, action, reward, observation, done);

                    LastObservation = observation;

                    if (done)
                        break;
                }

                OnEpisodeEnd(ep);

                if (Verbose)
                    Console.WriteLine($"Ep {ep}: reward {Math.Round(totalReward, 2)} epsilon {Math.Round(Epsilon, 4)}");

                Epsilon = Math.Max(MinEpsilon, Epsilon * EpsilonDecay);
            }
        }

        public double Test(int episodes, int maxStepsPerEpisode, bool render)
        {
            double[] totalRewards = new double[episodes];

            for (int ep = 0; ep < episodes; ++ep)
            {
                LastObservation = Env.Reset();
                double totalReward = 0;

                for (int step = 0; step < maxStepsPerEpisode; ++step)
                {
                    Tensor action = GetOptimalAction();

                    bool done = Env.Step(action, out var observation, out var reward);

                    LastObservation = observation;
                    totalReward += reward;

                    Render(render);

                    if (done)
                        break;
                }

                totalRewards[ep] = totalReward;

                if (Verbose)
                    Console.WriteLine($"Ep {ep}: reward {Math.Round(totalReward, 2)}");
            }

            return totalRewards.Sum() / episodes;
        }

        protected abstract Tensor GetOptimalAction();
        protected abstract void OnStep(int step, Tensor action, double reward, Tensor nextState, bool done);
        protected virtual void OnEpisodeEnd(int episode) { }
        protected virtual void Save() { }
        protected virtual void Load(int episode) { }

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
        public double MinEpsilon = 0.01;
        public double MaxEpsilon = 1.0;
        public double EpsilonDecay = 0.995;

        public bool Verbose = false;
        public int StepsPerSec = 30;

        public int SaveFrequency = 50; // save parameters every SaveFrequency episodes
    }
}
