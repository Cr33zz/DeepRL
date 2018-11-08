using System;
using System.Collections.Generic;
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
            if (env.ActionSpace.Shape.Length > 1)
                throw new Exception("Action space can contain only single value.");

            if (env.ObservationSpace.Shape.Length > 1)
                throw new Exception("Observation space can contain only single value.");

            ValueFuncModel = valueFuncModel;
        }

        public override void Train(int episodes, int maxStepsPerEpisode)
        {
            for (int ep = 0; ep < episodes; ++ep)
            {
                LastObservation = Env.Reset();

                for (int step = 0; step < maxStepsPerEpisode; ++step)
                {
                    int action = -1;

                    if (GlobalRandom.Rng.NextDouble() < Epsilon)
                        action = (int)Env.ActionSpace.Sample()[0]; // explore
                    else
                        action = (int)ValueFuncModel.GetOptimalAction(LastObservation); // exploit

                    bool done = Env.Step(action, out var observation, out var reward);

                    ValueFuncModel.OnTransition(LastObservation, action, reward, observation);

                    LastObservation = observation;

                    if (done)
                        break;
                }

                Epsilon = MinEpsilon + (MaxEpsilon - MinEpsilon) * Math.Exp(EpsilonDecay * ep);
            }
        }

        public override double Test(int episodes, int maxStepsPerEpisode)
        {
            double[] totalRewards = new double[episodes];

            for (int ep = 0; ep < episodes; ++ep)
            {
                LastObservation = Env.Reset();
                double totalReward = 0;

                for (int step = 0; step < maxStepsPerEpisode; ++step)
                {
                    int action = (int)ValueFuncModel.GetOptimalAction(LastObservation);

                    bool done = Env.Step(action, out var observation, out var reward);

                    LastObservation = observation;
                    totalReward += reward;

                    if (Verbose)
                        Env.Render();

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
