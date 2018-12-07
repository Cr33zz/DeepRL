using System;
using DeepQL.Agents;
using DeepQL.Environments;
using DeepQL.Gyms;
using DeepQL.ValueFunc;
using Neuro.Tensors;

namespace Examples
{
    class LunarLander
    {
        static void Main(string[] args)
        {
            //Tensor.SetOpMode(Tensor.OpMode.GPU);

            Env env = new LunarLanderEnv();

            var qFunc = new DQN(env.ObservationSpace.Shape, env.ActionSpace.NumberOfValues(), new[]{ 128, 64 },  0.0001f, 0.999f, 10000)
            {
                BatchSize = 32,
                MemoryInterval = 1,
                TargetModelUpdateInterval = 400,
                //TargetModelUpdateOnEpisodeEnd = true,
                TrainingEpochs = 1
            };

            Agent agent = new AgentQL("dqn_lunarlander", env, qFunc)
            {
                WarmupSteps = 1000,
                MaxEpsilon = 1.0f,
                EpsilonDecay = 0.99f,
                TrainInterval = 1,
                RewardClipping = false,
                TrainRenderInterval = 10,
                Verbose = true,
                RenderFreq = 50,
            };

            //agent.Load($"{agent.Name}_500");
            agent.Train(1500, 2000);

            Console.WriteLine($"Average reward {agent.Test(100, 2000, true)}");
            return;
        }
    }
}
