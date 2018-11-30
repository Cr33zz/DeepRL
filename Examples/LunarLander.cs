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
                TargetModelUpdateFreq = 1,
                TrainingEpochs = 2
            };

            Agent agent = new AgentQL("dqn_lunarlander", env, qFunc)
            {
                StepsBeforeTraining = 2000,
                MaxEpsilon = 0.7f,
                EpsilonDecay = 0.98f,
                //TrainOnlyOnEpisodeEnd = true,
                Verbose = true,
                TrainingRenderFreq = 20
            };

            agent.Train(500, 5000);

            Console.WriteLine($"Average reward {agent.Test(10, 1000, true)}");
        }
    }
}
