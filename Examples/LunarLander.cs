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
            //Tensor.SetOpMode(Tensor.OpMode.GPU);

            Env env = new LunarLanderEnv();

            var qFunc = new DQN(env.ObservationSpace.Shape, env.ActionSpace.NumberOfValues(), new[]{ 128, 64 },  0.0001f, 0.999f, 32, new PriorityExperienceReplay(100000))
            {
                MemoryInterval = 1,
                EnableDoubleDQN = true,
                TargetModelUpdateInterval = 4000,
                //TargetModelUpdateOnEpisodeEnd = true,
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
                RenderFreq = 50,
            };

            //agent.Load($"{agent.Name}_1500");

            agent.Train(1500, 2000);

            Console.WriteLine($"Average reward {agent.Test(100, 1000, true)}");
            return;
        }
    }
}
