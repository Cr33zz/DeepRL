using DeepQL.Spaces;
using Neuro.Tensors;

namespace DeepQL.Environments
{
    public abstract class Env
    {
        protected Env(Space actionSpace, Space observationSpace)
        {
            ActionSpace = actionSpace;
            ObservationSpace = observationSpace;
        }

        // Returns true when end state has been reached
        public abstract bool Step(int action, out Tensor observation, out double reward);
        public abstract Tensor Reset();
        public abstract void Render();
        public abstract void Seed(int seed = 0);

        public readonly Space ActionSpace;
        public readonly Space ObservationSpace;
    }
}
