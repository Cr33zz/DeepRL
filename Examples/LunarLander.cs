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

            Agent agent = new AgentQL("dqn_lunarlander", env, new DQN(env.ObservationSpace.Shape, env.ActionSpace.NumberOfValues(), 0.001f, 0.98f, 50000) { BatchSize = 32 })
            {
                MaxEpsilon = 0.8f,
                //RewardOnDone = -50,
                Verbose = true,
                //EpsilonDecay = 0.999f
            };
            agent.Train(1000, 2000, false);

            Console.WriteLine($"Average reward {agent.Test(10, 2000, true)}");
        }
    }
}
