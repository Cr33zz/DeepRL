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
            Env env = new AtariEnv("../../../roms/pong.bin", true);

			var memory = new PriorityExperienceReplay(100000);

			var qFunc = new DQNConv(new []{64,64}, env.ActionSpace.NumberOfValues(), 0.00025f, 0.999f, 32, memory)
			{
				MemoryInterval = 1,
				EnableDoubleDQN = true,
				TargetModelUpdateInterval = 4000,
				TrainingEpochs = 1
			};

			Agent agent = new AgentQL("dqnconv_atari", env, qFunc)
			{
				WarmupSteps = 1000,
				MaxEpsilon = 1.0f,
				MinEpsilon = 0.01f,
				EpsilonDecay = 0.995f,
				TrainInterval = 4,
				RewardClipping = false,
				TrainRenderInterval = 1,
				Verbose = true,
				RenderFreq = 60,
			};

			agent.Train(1500, 1000);

			return;
        }
    }
}
