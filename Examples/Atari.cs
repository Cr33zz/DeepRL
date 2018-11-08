using System.Threading;
using DeepQL.Agents;
using DeepQL.Environments;
using DeepQL.Gyms;

namespace Examples
{
    class Atari
    {
        static void Main(string[] args)
        {
            Env env = new AtariEnv("../../../roms/breakout.bin");

            while (!env.Step(env.ActionSpace.Sample(), out var nextState, out var reward))
            {
                env.Render();
                Thread.Sleep(15);
            }

            return;
        }
    }
}
