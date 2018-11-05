using System;
using System.Collections.Generic;
using System.Linq;
using DeepQL.Environments;
using Neuro.Tensors;

namespace DeepQL.Agents
{
    public class AgentQTable : Agent
    {
        public AgentQTable(Env env, double learningRate, double discoutFactor, bool verbose = false)
            : base(env, learningRate, discoutFactor, verbose)
        {
            if (env.ActionSpace.Shape.Length > 1)
                throw new Exception("Action space can contain only single value.");

            if (env.ObservationSpace.Shape.Length > 1)
                throw new Exception("Observation space can contain only single value.");

            QTable = new double[env.ObservationSpace.NumberOfValues(), env.ActionSpace.NumberOfValues()];
        }

        public override void Train(int episodes, int maxStepsPerEpisode)
        {
            for (int ep = 0; ep < episodes; ++ep)
            {
                State = Env.Reset();

                for (int step = 0; step < maxStepsPerEpisode; ++step)
                {
                    int state = (int)State[0];
                    int action = -1;

                    if (GlobalRandom.Rng.NextDouble() < Epsilon)
                        action = (int)Env.ActionSpace.Sample()[0]; // explore
                    else
                        action = BestActionBasedOnQTable(state); // exploit

                    bool done = Env.Step(action, out var nextState, out var reward);

                    QTable[state, action] += LearningRate * (reward + DiscountFactor * GetMaxRewardBasedOnQTable((int)nextState[0]) - QTable[state, action]);

                    State = nextState;

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
                State = Env.Reset();
                double totalReward = 0;

                for (int step = 0; step < maxStepsPerEpisode; ++step)
                {
                    int state = (int)State[0];
                    int action = BestActionBasedOnQTable(state);

                    bool done = Env.Step(action, out var nextState, out var reward);

                    State = nextState;
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

        private int BestActionBasedOnQTable(int state)
        {
            double max = double.MinValue;
            int bestA = -1;

            for (int a = 0; a < QTable.GetLength(1); ++a)
            {
                if (QTable[state, a] > max)
                {
                    max = QTable[state, a];
                    bestA = a;
                }
            }

            return bestA;
        }

        private double GetMaxRewardBasedOnQTable(int state)
        {
            double max = double.MinValue;
            
            for (int a = 0; a < QTable.GetLength(1); ++a)
                max = Math.Max(QTable[state, a], max);

            return max;
        }

        private readonly double[,] QTable;
    }
}
