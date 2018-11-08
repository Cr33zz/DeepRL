using System.Threading;
using DeepQL.Agents;
using DeepQL.Environments;
using DeepQL.Gyms;

namespace Examples
{
    class Pendulum
    {
        static void Main(string[] args)
        {
            Env env = new PendulumEnv();

            while (!env.Step(env.ActionSpace.Sample(), out var nextState, out var reward))
            {
                env.Render();
                Thread.Sleep(50);
            }

            return;
        }
    }
}
