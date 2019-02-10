using System.Threading;
using DeepQL.Agents;
using DeepQL.Environments;
using DeepQL.Gyms;
using DeepQL.MemoryReplays;
using DeepQL.ValueFunc;

namespace Examples
{
    class Atari
    {
        static void Main(string[] args)
        {
            Env env = new AtariEnv("../../../roms/breakout.bin", true);

			var memory = new PriorityExperienceReplay(100000);

			var qFunc = new DQNConv(env.ActionSpace.NumberOfValues(), 0.00025f, 0.999f, 32, memory)
			{
				MemoryInterval = 1,
				EnableDoubleDQN = true,
				TargetModelUpdateInterval = 4000,
				TrainingEpochs = 1
			};

			Agent agent = new AgentQL("dqnconv_breakout", env, qFunc)
			{
				WarmupSteps = 5000,
				MaxEpsilon = 1.0f,
				MinEpsilon = 0.01f,
				EpsilonDecay = 0.995f,
				TrainInterval = 1,
				RewardClipping = false,
				TrainRenderInterval = 10,
				Verbose = true,
				RenderFreq = 60,
			};

			agent.Train(1500, 15000);

			return;
        }
    }
}
