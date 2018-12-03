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

            var qFunc = new DQN(env.ObservationSpace.Shape, env.ActionSpace.NumberOfValues(), new[]{ 128, 64 },  0.001f, 0.99f, 60000)
            {
                BatchSize = 32,
                MemoryInterval = 1,
                TargetModelUpdateInterval = 50,
                TrainingEpochs = 2
            };

            Agent agent = new AgentQL("dqn_lunarlander", env, qFunc)
            {
                WarmupSteps = 1000,
                MaxEpsilon = 1.0f,
                EpsilonDecay = 0.99f,
                RewardClipping = false,
                TrainInterval = 2,
                Verbose = true,
                TrainRenderInterval = 10
            };

            //agent.Train(1500, 2000);

            agent.Load($"{agent.Name}_500");

            Console.WriteLine($"Average reward {agent.Test(10, 2000, true)}");
        }
    }
}
