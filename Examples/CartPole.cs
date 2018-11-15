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

            Agent agent = new AgentQL("dqn_cartpole", env, new DQN(env.ObservationSpace.Shape, env.ActionSpace.NumberOfValues(), 0.002, 0.8), true);
            agent.Train(500, 300, false);
            Console.WriteLine($"Average reward {agent.Test(10, 300, true)}");

            //while (!env.Step((int)env.ActionSpace.Sample()[0], out var nextState, out var reward))
            //{
            //    env.Render();
            //    Thread.Sleep(50);
            //}

            return;
        }
    }
}
