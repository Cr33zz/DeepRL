using System.Threading;
using DeepQL.Agents;
using DeepQL.Environments;
using DeepQL.Gyms;

namespace Examples
{
    class LunarLander
    {
        static void Main(string[] args)
        {
            Env env = new LunarLanderEnv();

            while (!env.Step(env.ActionSpace.Sample(), out var nextState, out var reward))
            {
                env.Render();
                Thread.Sleep(15);
            }

            return;
        }
    }
}
