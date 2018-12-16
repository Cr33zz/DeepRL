using System;
using DeepQL.Agents;
using DeepQL.Environments;
using DeepQL.Gyms;
using DeepQL.ValueFunc;
using DeepQL.MemoryReplays;

namespace Examples
{
    class LunarLander
    {
        static void Main(string[] args)
        {
            Env env = new LunarLanderEnv();

            var memory = new PriorityExperienceReplay(100000);

            var qFunc = new DQN(env.ObservationSpace.Shape, env.ActionSpace.NumberOfValues(), new[]{ 256, 128 },  0.0001f, 0.999f, 32, memory)
            {
                MemoryInterval = 1,
                EnableDoubleDQN = true,
                TargetModelUpdateInterval = 4000,
                TrainingEpochs = 1
            };

            Agent agent = new AgentQL("dqn_lunarlander", env, qFunc)
            {
                WarmupSteps = 5000,
                MaxEpsilon = 1.0f,
                MinEpsilon = 0.01f,
                EpsilonDecay = 0.995f,
                TrainInterval = 1,
                RewardClipping = false,
                TrainRenderInterval = 10,
                Verbose = true,
                RenderFreq = 80,
            };

            //agent.Train(1500, 1500);
            agent.Load($"{agent.Name}_1500");
            agent.Test(100, 400, 2);
        }
    }
}
