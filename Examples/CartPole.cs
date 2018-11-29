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
            //https://github.com/rlcode/reinforcement-learning/blob/master/2-cartpole/1-dqn/cartpole_dqn.py
            Agent agent = new AgentQL("dqn_cartpole", env, new DQN(env.ObservationSpace.Shape, env.ActionSpace.NumberOfValues(), 0.001f, 0.99f, 2000) { BatchSize = 64, TargetModelUpdateFreq = 1 })
            {
                Verbose = true,
                RewardOnDone = -100,
                EpsilonDecayMode = EEpsilonDecayMode.EveryStep,
                EpsilonDecay = 0.999f,
                StepsBeforeTraining = 1000
            };
            agent.Train(300, 500);
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
