using System;
using System.Linq;
using DeepQL.Environments;
using DeepQL.ValueFunc;
using Neuro.Tensors;

namespace DeepQL.Agents
{
    public class AgentQL : Agent
    {
        public AgentQL(Env env, ValueFunctionModel valueFuncModel, bool verbose = false)
            : base(env, verbose)
        {
            ValueFuncModel = valueFuncModel;
        }

        public override void Train(int episodes, int maxStepsPerEpisode, bool render)
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
                        action = ValueFuncModel.GetOptimalAction(LastObservation); // exploit

                    bool done = Env.Step(action, out var observation, out var reward);
                    totalReward += reward;

                    Render(render);

                    ValueFuncModel.OnTransition(LastObservation, action, reward, observation, done);

                    LastObservation = observation;

                    if (done)
                        break;
                }

                if (Verbose)
                    Console.WriteLine($"Ep {ep}: reward {Math.Round(totalReward, 2)} epsilon {Math.Round(Epsilon, 4)}");

                Epsilon = Math.Max(MinEpsilon, Epsilon * EpsilonDecay);
            }
        }

        public override double Test(int episodes, int maxStepsPerEpisode, bool render)
        {
            double[] totalRewards = new double[episodes];

            for (int ep = 0; ep < episodes; ++ep)
            {
                LastObservation = Env.Reset();
                double totalReward = 0;

                for (int step = 0; step < maxStepsPerEpisode; ++step)
                {
                    Tensor action = ValueFuncModel.GetOptimalAction(LastObservation);

                    bool done = Env.Step(action, out var observation, out var reward);

                    LastObservation = observation;
                    totalReward += reward;

                    Render(render);

                    if (done)
                        break;
                }

                totalRewards[ep] = totalReward;

                if (Verbose)
                    Console.WriteLine($"Episode {ep} total reward {totalReward}");
            }

            return totalRewards.Sum() / episodes;
        }

        protected override int ChooseNextAction()
        {
            throw new NotImplementedException();
        }

        protected override void OnStep(Tensor state, int action, double reward, Tensor nextState)
        {
            throw new NotImplementedException();
        }

        private readonly ValueFunctionModel ValueFuncModel;
    }
}
