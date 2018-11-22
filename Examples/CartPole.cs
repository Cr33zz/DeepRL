using System;
using DeepQL.Agents;
using DeepQL.Environments;
using DeepQL.Gyms;
using DeepQL.ValueFunc;

namespace Examples
{
    class CartPole
    {
        static void Main(string[] args)
        {
            Env env = new CartPoleEnv();

            Agent agent = new AgentQL("dqn_cartpole", env, new DQN(env.ObservationSpace.Shape, env.ActionSpace.NumberOfValues(), 0.05, 0.99, 2000) { BatchSize = 32 }) { /*RewardOnDone = -50,*/ Verbose = true };
            agent.Train(500, 300, false);
            Console.WriteLine($"Average reward {agent.Test(50, 300, false)}");

            //while (!env.Step((int)env.ActionSpace.Sample()[0], out var nextState, out var reward))
            //{
            //    env.Render();
            //    Thread.Sleep(50);
            //}

            return;
        }
    }
}
