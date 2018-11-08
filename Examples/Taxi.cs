using System;
using DeepQL.Agents;
using DeepQL.Environments;
using DeepQL.Gyms;
using DeepQL.ValueFunc;

namespace Examples
{
    class Taxi
    {
        static void Main(string[] args)
        {
            Env env = new TaxiEnv();

            Agent agent = new AgentQL(env, new QTable(env.ObservationSpace.NumberOfValues(), env.ActionSpace.NumberOfValues(), 0.7, 0.618), true);
            agent.Train(50000, 100);
            Console.WriteLine($"Average reward {agent.Test(100, 100)}");

            //while (!env.Step((int)env.ActionSpace.Sample()[0], out var nextState, out var reward))
            //    env.Render();

            return;
        }
    }
}
