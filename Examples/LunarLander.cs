using System;
using DeepQL.Agents;
using DeepQL.Environments;
using DeepQL.Gyms;
using DeepQL.ValueFunc;

namespace Examples
{
    class LunarLander
    {
        static void Main(string[] args)
        {
            Env env = new LunarLanderEnv();

            var qFunc = new DQN(env.ObservationSpace.Shape, env.ActionSpace.NumberOfValues(), 0.001f, 0.95f, 60000)
            {
                BatchSize = 32,
                MemoryInterval = 1,
                TargetModelUpdateInterval = 100,
                TrainingEpochs = 2
            };

            Agent agent = new AgentQL("dqn_lunarlander", env, qFunc)
            {
                WarmupSteps = 2000,
                MaxEpsilon = 0.9f,
                EpsilonDecay = 0.99f,
                RewardClipping = false,
                TrainInterval = 4,
                Verbose = true,
                TrainRenderInterval = 5
            };

            agent.Train(500, 2000);

            Console.WriteLine($"Average reward {agent.Test(10, 2000, true)}");
        }
    }
}
