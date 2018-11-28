﻿using System;
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
            //https://github.com/rlcode/reinforcement-learning/blob/master/2-cartpole/1-dqn/cartpole_dqn.py
            Agent agent = new AgentQL("dqn_cartpole", env, new DQN(env.ObservationSpace.Shape, env.ActionSpace.NumberOfValues(), 0.001f, 0.99f, 2000) { BatchSize = 64 })
            {
                //RewardOnDone = -50,
                Verbose = true,
                //EpsilonDecay = 0.999f
            };
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
