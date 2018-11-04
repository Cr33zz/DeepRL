using DeepQL.Environments;
using Neuro.Tensors;

namespace ExampleTaxi
{
    class Program
    {
        static void Main(string[] args)
        {
            Env env = new TaxiEnv();
            env.Reset();

            Tensor nextState;
            double reward;

            while (!env.Step((int)env.ActionSpace.Sample()[0], out nextState, out reward))
                env.Render();

            return;
        }
    }
}
